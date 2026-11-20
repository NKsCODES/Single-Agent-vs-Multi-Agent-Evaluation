// Azure deployment for the agent architecture evaluation service.
//
// Creates: Log Analytics + Application Insights, a user-assigned managed identity, Azure
// Container Registry, Azure OpenAI (large, small and embedding deployments), Azure Cosmos DB
// for NoSQL (serverless, vector + full-text search, container with the vector, full-text and
// indexing policies), Azure AI Content Safety, and a Container Apps environment and app.
//
// Every service is reached with the managed identity (Entra ID). Local auth (keys) is disabled
// on Azure OpenAI, Content Safety and Cosmos DB, so there are no secrets to store or rotate.

targetScope = 'resourceGroup'

@description('Short prefix for resource names (3 to 10 lowercase letters or digits).')
@minLength(3)
@maxLength(10)
param prefix string = 'archeval'

param location string = resourceGroup().location

@description('Container image, for example myacr.azurecr.io/archeval:sha. Leave empty on the first deploy.')
param image string = ''

@description('Large model for the lead agent and single-large. Pick a model available in your region.')
param largeModel object = { name: 'gpt-4.1', version: '2025-04-14', capacity: 50 }

@description('Small model for workers and single-small.')
param smallModel object = { name: 'gpt-4.1-mini', version: '2025-04-14', capacity: 100 }

param embeddingModel object = { name: 'text-embedding-3-small', version: '1', capacity: 50 }

@description('Embedding dimensions stored in Cosmos DB. Must match ARCHEVAL_EMBEDDING_DIMENSIONS.')
param embeddingDimensions int = 1536

param minReplicas int = 1
param maxReplicas int = 5

var suffix = uniqueString(resourceGroup().id, prefix)
var tags = { app: 'agent-architecture-eval' }

// ---------------------------------------------------------------- observability

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs-${suffix}'
  location: location
  tags: tags
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-ai-${suffix}'
  location: location
  kind: 'web'
  tags: tags
  properties: { Application_Type: 'web', WorkspaceResourceId: logs.id, DisableLocalAuth: false }
}

// ---------------------------------------------------------------- identity

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-id-${suffix}'
  location: location
  tags: tags
}

// ---------------------------------------------------------------- registry

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${prefix}acr${suffix}'
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: acr
  name: guid(acr.id, identity.id, 'acrpull')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ---------------------------------------------------------------- Azure OpenAI

resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: '${prefix}-aoai-${suffix}'
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: '${prefix}-aoai-${suffix}'
    disableLocalAuth: true
    publicNetworkAccess: 'Enabled'
  }
}

resource largeDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: 'large'
  sku: { name: 'GlobalStandard', capacity: largeModel.capacity }
  properties: { model: { format: 'OpenAI', name: largeModel.name, version: largeModel.version } }
}

resource smallDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: 'small'
  dependsOn: [ largeDeployment ]
  sku: { name: 'GlobalStandard', capacity: smallModel.capacity }
  properties: { model: { format: 'OpenAI', name: smallModel.name, version: smallModel.version } }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: 'embedding'
  dependsOn: [ smallDeployment ]
  sku: { name: 'Standard', capacity: embeddingModel.capacity }
  properties: { model: { format: 'OpenAI', name: embeddingModel.name, version: embeddingModel.version } }
}

resource openaiUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: openai
  name: guid(openai.id, identity.id, 'openai-user')
  properties: {
    // Cognitive Services OpenAI User
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ---------------------------------------------------------------- Content Safety (Prompt Shields)

resource contentSafety 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: '${prefix}-cs-${suffix}'
  location: location
  tags: tags
  kind: 'ContentSafety'
  sku: { name: 'S0' }
  properties: { customSubDomainName: '${prefix}-cs-${suffix}', disableLocalAuth: true }
}

resource contentSafetyUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: contentSafety
  name: guid(contentSafety.id, identity.id, 'cs-user')
  properties: {
    // Cognitive Services User
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'a97b65f3-24c7-4388-baec-2e87135dc908')
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ---------------------------------------------------------------- Cosmos DB for NoSQL

resource cosmos 'Microsoft.DocumentDB/databaseAccounts@2025-10-15' = {
  name: '${prefix}-cosmos-${suffix}'
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    locations: [ { locationName: location, failoverPriority: 0 } ]
    consistencyPolicy: { defaultConsistencyLevel: 'Session' }
    capabilities: [
      { name: 'EnableServerless' }
      { name: 'EnableNoSQLVectorSearch' }
      { name: 'EnableNoSQLFullTextSearch' }
    ]
    disableLocalAuth: true
    minimalTlsVersion: 'Tls12'
  }
}

resource database 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2025-10-15' = {
  parent: cosmos
  name: 'archeval'
  properties: { resource: { id: 'archeval' } }
}

resource chunks 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2025-10-15' = {
  parent: database
  name: 'chunks'
  properties: {
    resource: {
      id: 'chunks'
      partitionKey: { paths: [ '/doc_id' ], kind: 'Hash' }
      vectorEmbeddingPolicy: {
        vectorEmbeddings: [ { path: '/embedding', dataType: 'float32', distanceFunction: 'cosine', dimensions: embeddingDimensions } ]
      }
      fullTextPolicy: {
        defaultLanguage: 'en-US'
        fullTextPaths: [ { path: '/text', language: 'en-US' } ]
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        automatic: true
        includedPaths: [ { path: '/*' } ]
        excludedPaths: [ { path: '/embedding/*' }, { path: '/"_etag"/?' } ]
        vectorIndexes: [ { path: '/embedding', type: 'diskANN' } ]
        fullTextIndexes: [ { path: '/text' } ]
      }
    }
  }
}

resource cosmosDataContributor 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2025-10-15' = {
  parent: cosmos
  name: guid(cosmos.id, identity.id, 'data-contributor')
  properties: {
    // Cosmos DB Built-in Data Contributor (data plane)
    roleDefinitionId: '${cosmos.id}/sqlRoleDefinitions/00000000-0000-0000-0000-000000000002'
    principalId: identity.properties.principalId
    scope: cosmos.id
  }
}

// ---------------------------------------------------------------- Container Apps

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env-${suffix}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: { customerId: logs.properties.customerId, sharedKey: logs.listKeys().primarySharedKey }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-api'
  location: location
  tags: tags
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${identity.id}': {} } }
  dependsOn: [ acrPull, openaiUser, contentSafetyUser, cosmosDataContributor ]
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: { external: true, targetPort: 8000, transport: 'auto' }
      registries: empty(image) ? [] : [ { server: acr.properties.loginServer, identity: identity.id } ]
      secrets: [ { name: 'appinsights', value: appInsights.properties.ConnectionString } ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: empty(image) ? 'mcr.microsoft.com/k8se/quickstart:latest' : image
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: [
            { name: 'ARCHEVAL_ENV', value: 'prod' }
            { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
            { name: 'ARCHEVAL_LLM_PROVIDER', value: 'azure' }
            { name: 'ARCHEVAL_AZURE_OPENAI_ENDPOINT', value: openai.properties.endpoint }
            { name: 'ARCHEVAL_LARGE_DEPLOYMENT', value: largeDeployment.name }
            { name: 'ARCHEVAL_SMALL_DEPLOYMENT', value: smallDeployment.name }
            { name: 'ARCHEVAL_EMBEDDINGS', value: 'azure' }
            { name: 'ARCHEVAL_EMBEDDING_DEPLOYMENT', value: embeddingDeployment.name }
            { name: 'ARCHEVAL_EMBEDDING_DIMENSIONS', value: string(embeddingDimensions) }
            { name: 'ARCHEVAL_VECTOR_STORE', value: 'cosmos' }
            { name: 'ARCHEVAL_COSMOS_ENDPOINT', value: cosmos.properties.documentEndpoint }
            { name: 'ARCHEVAL_COSMOS_DATABASE', value: database.name }
            { name: 'ARCHEVAL_COSMOS_CONTAINER', value: chunks.name }
            { name: 'ARCHEVAL_CONTENT_SAFETY_ENDPOINT', value: contentSafety.properties.endpoint }
            { name: 'ARCHEVAL_APPLICATIONINSIGHTS_CONNECTION_STRING', secretRef: 'appinsights' }
          ]
          probes: [
            { type: 'Liveness', httpGet: { path: '/healthz', port: 8000 }, periodSeconds: 15 }
            { type: 'Readiness', httpGet: { path: '/readyz', port: 8000 }, periodSeconds: 10, failureThreshold: 3 }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [ { name: 'http', http: { metadata: { concurrentRequests: '20' } } } ]
      }
    }
  }
}

output apiUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
output acrLoginServer string = acr.properties.loginServer
output openaiEndpoint string = openai.properties.endpoint
output cosmosEndpoint string = cosmos.properties.documentEndpoint
output identityClientId string = identity.properties.clientId
