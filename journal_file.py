import matplotlib.pyplot as plt
import numpy as np
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

get_generic_pdk().activate()

processed_gds = "output/lionix_gate/chip_design_merged_flattened.gds"
journal_filename = 'output/lionix_gate/chip_geometry.jou'
abaqus_filename = 'run_file/chip_geometry.inp'
target_layer = (1, 0)

component = gf.import_gds(processed_gds)

# dictionary of polygon vertices indexed by layers
polygons_by_layer = component.get_polygons_points(by="tuple", layers=[target_layer])

# Define the layer of interest
target_layers = list(polygons_by_layer.keys())


# Retrieve polygons from target layers
tol_polygons = 0
polygon_lst = []
tol_vertices = 0

for key in target_layers:
    target_polygons = polygons_by_layer.get(key, [])
    tol_polygons += len(target_polygons)
    for polygon in target_polygons:
        polygon_lst.append(np.array(polygon))
        tol_vertices += len(polygon)

f = open(journal_filename, 'w') # Pay attention to the mode
f.write('reset\n\n')

temp = 0
for gon in polygon_lst:
    # create vertex
    for v in gon:
        f.write('create vertex ' + str(v[0]) + ' ' + str(v[1]) + '\n')
    # create surface
    f.write('\n')
    vid_lst = [temp + i + 1 for i in range(len(gon))]
    f.write('create surface vertex ')
    for i in vid_lst:
        f.write(str(i) + ' ')
    f.write('\n\n')
    temp += len(gon)

# Mesh control
f.write('# {gap=20}\n')
f.write('curve all scheme equal\n')
f.write('curve  all  except  curve  4  size  {gap/2}\n')
f.write('curve  4  size  {4*gap}\n')
f.write('trimesher  surface  gradation  {1.4}\n')
f.write('surface  all  scheme  trimesh\n')
f.write('mesh  surface  all\n')
f.write('imprint  all\n')
f.write('merge  all\n\n')

for i in range(tol_polygons):
    f.write('block ' + str(i + 1) + ' surface ' + str(i + 1) + '\n')

f.write('compress node\n')
f.write('compress tri\n\n')

f.write(f'export abaqus "{abaqus_filename}" block all dimension 3 overwrite')
f.close()

# Plot polygons — click a surface to pin its index label
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath

fig, ax = plt.subplots(figsize=(10, 10))
patches = []
for i, pts in enumerate(polygon_lst):
    patch = MplPolygon(pts, closed=True, edgecolor="steelblue", facecolor="lightblue", linewidth=0.5)
    ax.add_patch(patch)
    patches.append((patch, i + 1, pts))

ax.autoscale()
ax.set_aspect("equal")
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
ax.set_title("Merged polygons — layer (1,0)  |  click a polygon to label it")

click_labels = []

def on_click(event):
    if event.inaxes is not ax or event.xdata is None:
        return
    for patch, idx, pts in patches:
        if MplPath(np.vstack([pts, pts[0]])).contains_point((event.xdata, event.ydata)):
            lbl = ax.text(event.xdata, event.ydata, str(idx), ha="center", va="center",
                          fontsize=6, color="black",
                          bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8))
            click_labels.append(lbl)
            patch.set_facecolor("orange")
            fig.canvas.draw_idle()
            break

fig.canvas.mpl_connect("button_press_event", on_click)

plt.tight_layout()
#plt.savefig("output/lionx_gate/merged_polygons.png", dpi=200)
plt.show()