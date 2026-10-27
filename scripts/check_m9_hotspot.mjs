import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {gunzipSync} from 'node:zlib';
const root=(process.argv[2] || process.cwd()).replaceAll('\\', '/');
const html=fs.readFileSync(root+'/research/2026-10-27-m9-coupled/evidence/report/risk_hotspots.html','utf8');
const dataMatch=html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/);
assert.ok(dataMatch);const data=JSON.parse(dataMatch[1]);
const rawDemo=JSON.parse(gunzipSync(fs.readFileSync(root+'/research/2026-10-27-m9-coupled/evidence/demo.json.gz')).toString('utf8'));
for(const [target,source] of Object.entries({time:'time_s',truth:'severity_proxy',estimate:'predicted_severity_proxy',counts:'thermal_inspection_counts',temperature:'rack_temperature_c',predictedTemperature:'predicted_temperature_c',flow:'rack_flow_kg_s',age:'inspection_age_s',criticality:'criticality'}))assert.deepEqual(data[target],rawDemo[source]);
assert.deepEqual(data.stale,rawDemo.severity_proxy.map((row,frame)=>row.map((value,rack)=>value*Math.min(rawDemo.inspection_age_s[frame][rack]/300,4))));
const logic=html.slice(html.indexOf('</script><script>')+'</script><script>'.length,html.lastIndexOf('</script>'));
const panels=Object.fromEntries(['truth','estimate','counts','stale'].map(name=>[name,{children:[],append(cell){this.children.push(cell)}}]));
const objects={...panels,data:{textContent:dataMatch[1]},time:{value:'179',max:'359'},clock:{textContent:''},detail:{innerHTML:''}};
const document={getElementById:id=>{assert.ok(id in objects);return objects[id]},createElement:tag=>({tag,className:'',dataset:{},style:{},attributes:{},setAttribute(k,v){this.attributes[k]=v},classList:{toggle(){}}}),querySelectorAll:selector=>panels[selector.slice(1).split(' ')[0]].children};
const context=vm.createContext({document,Math,JSON,console});vm.runInContext(logic,context,{timeout:5000});
assert.ok(objects.clock.textContent.startsWith('900 s'));
for(const panel of Object.values(panels)){assert.equal(panel.children.length,128);assert.ok(panel.children.every(cell=>cell.style.background.startsWith('rgb(')))}
const frames=[0,59,60,179,239,240,359];
for(const frame of frames){objects.time.value=String(frame);objects.time.oninput();assert.ok(objects.clock.textContent.startsWith(data.time[frame].toFixed(0)+' s'));assert.ok(!objects.detail.innerHTML.includes('NaN'))}
panels.truth.children[127].onclick();assert.ok(objects.detail.innerHTML.includes('机柜 127'));assert.ok(objects.detail.innerHTML.includes('pod 15'));
objects.time.value='179';objects.time.oninput();panels.estimate.children[0].onclick();assert.ok(objects.detail.innerHTML.includes('机柜 0'));
const proof={offline_script_logic_passed:true,method:'Node VM with explicit minimal DOM stubs; no browser, UI or network action',frames_tested:frames,panels:4,cells_each:128,rack_selection_and_dependency_text_checked:[0,127],numeric_payload_matches_generated_file:true,browser_render_or_responsive_layout_verified:false,browser_preview_blocked_reason:'Browser URL policy rejects file protocol; local preview helper launch also rejected by policy',static_data_figures_visually_inspected:4};
fs.writeFileSync(root+'/research/2026-10-27-m9-coupled/evidence/hotspot-logic-verification.json',JSON.stringify(proof,null,2)+'\n');
process.stdout.write(JSON.stringify(proof,null,2)+'\n');
