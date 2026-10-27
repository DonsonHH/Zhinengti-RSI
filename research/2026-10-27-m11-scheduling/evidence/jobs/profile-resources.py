import json,importlib.util,sys,pathlib
from patrol_rsi.facility import FacilityConfig
root=pathlib.Path('/share/home/renly/renly-patrol-rsi/runs/20261027-m11')
spec=importlib.util.spec_from_file_location('patrol_rsi.profile_aisles',root/'aisles-profile.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
rows=[]
for base in ((12,2,2,2),(12,4,4,4),(16,2,2,2)):
 def counts(c,level):return {r:n*max(1,c.pods//16) for r,n in zip(m.ROLES,base)}
 m.fleet_counts=counts
 for racks,variant in ((128,'open'),(128,'restricted'),(3200,'open')):
  graph=m.AisleGraph(FacilityConfig(racks=racks),variant)
  row=m.feasibility(graph,'provisioned');row['base_counts']=base;rows.append(row)
  print(base,racks,variant,{r:max(x['closed_quantized_cycle_s'] for x in d['constructed_cycles']) for r,d in row['roles'].items()},flush=True)
with (root/'resource-profile.json').open('x') as h:json.dump(rows,h,indent=2)
