import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk
from pathlib import Path

def run_pre_process(
	design_file="lionix_gate/gds_file/chip_design.gds",
	output_file="lionix_gate/gds_file/chip_design_merged_flattened.gds",
	center_loc=((12920 + 13040) / 2, (5297.5 + 5502.5) / 2),
	electrode_layer=(37, 0),
	ito_layer=(12, 0),
	target_layer=(1, 0),
	show=False,
):
	from pathlib import Path
	
	get_generic_pdk().activate()
	
	project_root = Path(__file__).resolve().parents[1]
	design_path = Path(design_file)
	if not design_path.is_absolute():
		design_path = (project_root / design_path).resolve()

	output_path = Path(output_file)
	if not output_path.is_absolute():
		output_path = (project_root / output_path).resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)

	component = gf.import_gds(str(design_path))
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

	clean = gf.Component()
	for points in merged.get_polygons_points(by="tuple", layers=[target_layer]).get(target_layer, []):
		clean.add_polygon(points, layer=target_layer)

	clean.flatten()
	clean.write_gds(str(output_path), with_metadata=False, no_empty_cells=True)
	if show:
		try:
			rel_parent = design_path.relative_to(project_root).parent
		except ValueError:
			rel_parent = Path()
		(project_root / "build" / "oas" / rel_parent).mkdir(parents=True, exist_ok=True)
		clean.show()
	print(str(output_path))
	return str(output_path)


if __name__ == "__main__":
	run_pre_process(show=True)