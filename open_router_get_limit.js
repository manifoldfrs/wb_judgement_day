// open_router_get_limit.js
require('dotenv').config();

const reloadEnv = () => {
  delete require.cache[require.resolve('dotenv')];
  require('dotenv').config();
};

reloadEnv();

const apiKey = process.env.OPENROUTER_API_KEY;

// Use dynamic import for node-fetch
(async () => {
  const fetch = (await import('node-fetch')).default;

  fetch('https://openrouter.ai/api/v1/auth/key', {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
  })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error('Error:', error));
})();