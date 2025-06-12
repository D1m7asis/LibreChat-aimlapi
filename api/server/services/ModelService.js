const axios = require('axios');
const { Providers } = require('@librechat/agents');
const { HttpsProxyAgent } = require('https-proxy-agent');
const { EModelEndpoint, defaultModels, CacheKeys } = require('librechat-data-provider');
const { inputSchema, logAxiosError, extractBaseURL, processModelData } = require('~/utils');
const { OllamaClient } = require('~/app/clients/OllamaClient');
const { isUserProvided } = require('~/server/utils');
const getLogStores = require('~/cache/getLogStores');
const { logger } = require('~/config');

/**
 * Splits a string by commas and trims each resulting value.
 * @param {string} input - The input string to split.
 * @returns {string[]} An array of trimmed values.
 */
const splitAndTrim = (input) => {
  if (!input || typeof input !== 'string') {
    return [];
  }
  return input
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
};

const { openAIApiKey, userProvidedOpenAI } = require('./Config/EndpointService').config;

/**
 * Fetches OpenAI models from the specified base API path or Azure, based on the provided configuration.
 *
 * @param {Object} params - The parameters for fetching the models.
 * @param {Object} params.user - The user ID to send to the API.
 * @param {string} params.apiKey - The API key for authentication with the API.
 * @param {string} params.baseURL - The base path URL for the API.
 * @param {string} [params.name='OpenAI'] - The name of the API; defaults to 'OpenAI'.
 * @param {boolean} [params.azure=false] - Whether to fetch models from Azure.
 * @param {boolean} [params.userIdQuery=false] - Whether to send the user ID as a query parameter.
 * @param {boolean} [params.createTokenConfig=true] - Whether to create a token configuration from the API response.
 * @param {string} [params.tokenKey] - The cache key to save the token configuration. Uses `name` if omitted.
 * @returns {Promise<string[]>} A promise that resolves to an array of model identifiers.
 * @async
 */
const fetchModels = async ({
  user,
  apiKey,
  baseURL,
  name = EModelEndpoint.openAI,
  azure = false,
  userIdQuery = false,
  createTokenConfig = true,
  tokenKey,
}) => {
  let models = [];

  if (!baseURL && !azure) {
    return models;
  }

  if (!apiKey) {
    return models;
  }

  if (name && name.toLowerCase().startsWith(Providers.OLLAMA)) {
    return await OllamaClient.fetchModels(baseURL);
  }

  try {
    const options = {
      headers: {},
      timeout: 5000,
    };

    if (name === EModelEndpoint.anthropic) {
      options.headers = {
        'x-api-key': apiKey,
        'anthropic-version': process.env.ANTHROPIC_VERSION || '2023-06-01',
      };
    } else if (name === EModelEndpoint.aimlapi) {
      options.headers = {
        'X-Title': 'LibreChat',
      };
      options.headers.Authorization = `Bearer ${apiKey}`;
    } else {
      options.headers.Authorization = `Bearer ${apiKey}`;
    }

    if (process.env.PROXY) {
      options.httpsAgent = new HttpsProxyAgent(process.env.PROXY);
    }

    if (process.env.OPENAI_ORGANIZATION && baseURL.includes('openai')) {
      options.headers['OpenAI-Organization'] = process.env.OPENAI_ORGANIZATION;
    }

    const url = new URL(`${baseURL}${azure ? '' : '/models'}`);
    if (user && userIdQuery) {
      url.searchParams.append('user', user);
    }
    const res = await axios.get(url.toString(), options);

    /** @type {z.infer<typeof inputSchema>} */
    const input = res.data;

    const validationResult = inputSchema.safeParse(input);
    if (validationResult.success && createTokenConfig) {
      const endpointTokenConfig = processModelData(input);
      const cache = getLogStores(CacheKeys.TOKEN_CONFIG);
      await cache.set(tokenKey ?? name, endpointTokenConfig);
    }
    models = input.data.map((item) => item.id);
  } catch (error) {
    const logMessage = `Failed to fetch models from ${azure ? 'Azure ' : ''}${name} API`;
    logAxiosError({ message: logMessage, error });
  }

  return models;
};

/**
 * Fetches models from the specified API path or Azure, based on the provided options.
 * @async
 * @function
 * @param {object} opts - The options for fetching the models.
 * @param {string} opts.user - The user ID to send to the API.
 * @param {boolean} [opts.azure=false] - Whether to fetch models from Azure.
 * @param {boolean} [opts.assistants=false] - Whether to fetch models from Azure.
 * @param {boolean} [opts.plugins=false] - Whether to fetch models from the plugins.
 * @param {string[]} [_models=[]] - The models to use as a fallback.
 */
const fetchOpenAIModels = async (opts, _models = []) => {
  let models = _models.slice() ?? [];
  let apiKey = openAIApiKey;
  const openaiBaseURL = 'https://api.openai.com/v1';
  let baseURL = openaiBaseURL;
  let reverseProxyUrl = process.env.OPENAI_REVERSE_PROXY;

  if (opts.assistants && process.env.ASSISTANTS_BASE_URL) {
    reverseProxyUrl = process.env.ASSISTANTS_BASE_URL;
  } else if (opts.azure) {
    return models;
    // const azure = getAzureCredentials();
    // baseURL = (genAzureChatCompletion(azure))
    //   .split('/deployments')[0]
    //   .concat(`/models?api-version=${azure.azureOpenAIApiVersion}`);
    // apiKey = azureOpenAIApiKey;
  }

  if (reverseProxyUrl) {
    baseURL = extractBaseURL(reverseProxyUrl);
  }

  const modelsCache = getLogStores(CacheKeys.MODEL_QUERIES);

  const cachedModels = await modelsCache.get(baseURL);
  if (cachedModels) {
    return cachedModels;
  }

  if (baseURL || opts.azure) {
    models = await fetchModels({
      apiKey,
      baseURL,
      azure: opts.azure,
      user: opts.user,
      name: EModelEndpoint.openAI,
    });
  }

  if (models.length === 0) {
    return _models;
  }

  if (baseURL === openaiBaseURL) {
    const regex = /(text-davinci-003|gpt-|o\d+)/;
    const excludeRegex = /audio|realtime/;
    models = models.filter((model) => regex.test(model) && !excludeRegex.test(model));
    const instructModels = models.filter((model) => model.includes('instruct'));
    const otherModels = models.filter((model) => !model.includes('instruct'));
    models = otherModels.concat(instructModels);
  }

  await modelsCache.set(baseURL, models);
  return models;
};

/**
 * Loads the default models for the application.
 * @async
 * @function
 * @param {object} opts - The options for fetching the models.
 * @param {string} opts.user - The user ID to send to the API.
 * @param {boolean} [opts.azure=false] - Whether to fetch models from Azure.
 * @param {boolean} [opts.plugins=false] - Whether to fetch models for the plugins endpoint.
 * @param {boolean} [opts.assistants=false] - Whether to fetch models for the Assistants endpoint.
 */
const getOpenAIModels = async (opts) => {
  let models = defaultModels[EModelEndpoint.openAI];

  if (opts.assistants) {
    models = defaultModels[EModelEndpoint.assistants];
  } else if (opts.azure) {
    models = defaultModels[EModelEndpoint.azureAssistants];
  }

  if (opts.plugins) {
    models = models.filter(
      (model) =>
        !model.includes('text-davinci') &&
        !model.includes('instruct') &&
        !model.includes('0613') &&
        !model.includes('0314') &&
        !model.includes('0301'),
    );
  }

  let key;
  if (opts.assistants) {
    key = 'ASSISTANTS_MODELS';
  } else if (opts.azure) {
    key = 'AZURE_OPENAI_MODELS';
  } else if (opts.plugins) {
    key = 'PLUGIN_MODELS';
  } else {
    key = 'OPENAI_MODELS';
  }

  if (process.env[key]) {
    models = splitAndTrim(process.env[key]);
    return models;
  }

  if (userProvidedOpenAI) {
    return models;
  }

  return await fetchOpenAIModels(opts, models);
};

/**
 * Fetches models from the AI/ML API.
 * Filters for chat completion models and optionally caches token configs.
 * @async
 * @function
 * @param {object} opts - The options for fetching the models.
 * @param {string} opts.user - The user ID to send to the API.
 * @param {string} opts.apiKey - API key for the provider.
 * @param {string} opts.baseURL - Base URL of the provider.
 * @param {boolean} [opts.userIdQuery=false] - Include user ID as query param.
 * @param {boolean} [opts.createTokenConfig=true] - Whether to save token config.
 * @param {string} [opts.tokenKey] - Cache key for token configs.
 * @param {string[]} [_models=[]] - Fallback models.
 */
const fetchAimlapiModels = async (
  opts,
  _models = [],
) => {
  logAxiosError({ message: 'Fetching models from AI/ML API', error: null });
  let models = _models.slice() ?? [];
  const { user, apiKey, baseURL, userIdQuery = false, createTokenConfig = true, tokenKey } = opts;

  if (!apiKey || !baseURL) {
    return models;
  }

  const modelsCache = getLogStores(CacheKeys.MODEL_QUERIES);
  const cachedModels = await modelsCache.get(baseURL);
  if (cachedModels) {
    return cachedModels;
  }

  try {
    const options = {
      headers: { Authorization: `Bearer ${apiKey}` },
      timeout: 5000,
    };
    if (process.env.PROXY) {
      options.httpsAgent = new HttpsProxyAgent(process.env.PROXY);
    }

    const url = new URL(`${baseURL}/models`);
    if (user && userIdQuery) {
      url.searchParams.append('user', user);
    }

    const res = await axios.get(url.toString(), options);
    /** @type {z.infer<typeof inputSchema>} */
    const input = res.data;
    const data = Array.isArray(input.data) ? input.data : [];
    const filtered = data.filter((item) => item.type === 'chat_completion');

    if (createTokenConfig && inputSchema.safeParse(input).success) {
      const endpointTokenConfig = processModelData({ data: filtered });
      const cache = getLogStores(CacheKeys.TOKEN_CONFIG);
      await cache.set(tokenKey ?? 'aimlapi', endpointTokenConfig);
    }

    models = filtered.map((item) => item.id);
  } catch (error) {
    logAxiosError({ message: 'Failed to fetch models from AI/ML API', error });
  }

  if (models.length === 0) {
    return _models;
  }

  await modelsCache.set(baseURL, models);
  return models;
};

/**
 * Retrieves models for the AI/ML API provider.
 * @async
 * @function
 * @param {object} opts - Options including user, apiKey and baseURL.
 */
const getAimlapiModels = async (opts = {}) => {
  return await fetchAimlapiModels(opts, []);
};

const getChatGPTBrowserModels = () => {
  let models = ['text-davinci-002-render-sha', 'gpt-4'];
  if (process.env.CHATGPT_MODELS) {
    models = splitAndTrim(process.env.CHATGPT_MODELS);
  }

  return models;
};

/**
 * Fetches models from the Anthropic API.
 * @async
 * @function
 * @param {object} opts - The options for fetching the models.
 * @param {string} opts.user - The user ID to send to the API.
 * @param {string[]} [_models=[]] - The models to use as a fallback.
 */
const fetchAnthropicModels = async (opts, _models = []) => {
  let models = _models.slice() ?? [];
  let apiKey = process.env.ANTHROPIC_API_KEY;
  const anthropicBaseURL = 'https://api.anthropic.com/v1';
  let baseURL = anthropicBaseURL;
  let reverseProxyUrl = process.env.ANTHROPIC_REVERSE_PROXY;

  if (reverseProxyUrl) {
    baseURL = extractBaseURL(reverseProxyUrl);
  }

  if (!apiKey) {
    return models;
  }

  const modelsCache = getLogStores(CacheKeys.MODEL_QUERIES);

  const cachedModels = await modelsCache.get(baseURL);
  if (cachedModels) {
    return cachedModels;
  }

  if (baseURL) {
    models = await fetchModels({
      apiKey,
      baseURL,
      user: opts.user,
      name: EModelEndpoint.anthropic,
      tokenKey: EModelEndpoint.anthropic,
    });
  }

  if (models.length === 0) {
    return _models;
  }

  await modelsCache.set(baseURL, models);
  return models;
};

const getAnthropicModels = async (opts = {}) => {
  let models = defaultModels[EModelEndpoint.anthropic];
  if (process.env.ANTHROPIC_MODELS) {
    models = splitAndTrim(process.env.ANTHROPIC_MODELS);
    return models;
  }

  if (isUserProvided(process.env.ANTHROPIC_API_KEY)) {
    return models;
  }

  try {
    return await fetchAnthropicModels(opts, models);
  } catch (error) {
    logger.error('Error fetching Anthropic models:', error);
    return models;
  }
};

const getGoogleModels = () => {
  let models = defaultModels[EModelEndpoint.google];
  if (process.env.GOOGLE_MODELS) {
    models = splitAndTrim(process.env.GOOGLE_MODELS);
  }

  return models;
};

const getBedrockModels = () => {
  let models = defaultModels[EModelEndpoint.bedrock];
  if (process.env.BEDROCK_AWS_MODELS) {
    models = splitAndTrim(process.env.BEDROCK_AWS_MODELS);
  }

  return models;
};

module.exports = {
  fetchModels,
  fetchAimlapiModels,
  splitAndTrim,
  getOpenAIModels,
  getAimlapiModels,
  getBedrockModels,
  getChatGPTBrowserModels,
  getAnthropicModels,
  getGoogleModels,
};
