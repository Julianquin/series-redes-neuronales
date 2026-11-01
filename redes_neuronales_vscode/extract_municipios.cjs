// Regenera data/municipios.json y data/municipios_geo.json desde el paquete npm "spanish-cities-info" (ISC, datos INE).
// Uso:  npm install spanish-cities-info && node extract_municipios.cjs
const m = require('spanish-cities-info'); const fs = require('fs'); const c = m.getAllCities();
fs.writeFileSync('data/municipios.json', JSON.stringify(c.map(x => ({ n: x.name, p: x.province, c: x.community }))));
fs.writeFileSync('data/municipios_geo.json', JSON.stringify(c.map(x => ({ n: x.name, p: x.province, c: x.community, lat: x.latitude, lon: x.longitude }))));
console.log('municipios:', c.length);
