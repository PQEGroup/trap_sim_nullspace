import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

get_generic_pdk().activate()

design_file = "output/lionix_gate/chip_design.gds"
output_file = "output/lionix_gate/chip_design_merged_flattened.gds"
center_loc = ((12920 + 13040) / 2, (5297.5 + 5502.5) / 2)
electrode_layer = (37, 0)
ito_layer = (12, 0)
target_layer = (1, 0)

component = gf.import_gds(design_file)
component.flatten()
component.dmove((-center_loc[0], -center_loc[1]))

merged = gf.boolean(
	component,
	component,
	operation="or",
	layer=target_layer,
	layer1=electrode_layer,
	layer2=ito_layer,
)

clean = gf.Component("chip_design_merged_flattened")
for points in merged.get_polygons_points(by="tuple", layers=[target_layer]).get(target_layer, []):
	clean.add_polygon(points, layer=target_layer)

clean.write_gds(output_file, with_metadata=False, no_empty_cells=True)
print(output_file)

#clean.show()