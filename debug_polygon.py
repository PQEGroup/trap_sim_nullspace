#!/usr/bin/env python3
"""Debug script to visualize polygons and check for issues."""

import sys
import numpy as np
import matplotlib.pyplot as plt
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath

def check_polygon_validity(polygon, polygon_id):
    """Check if a polygon is valid (no self-intersections, proper winding)."""
    points = np.array(polygon)
    n = len(points)
    
    issues = []
    
    # Check for duplicate consecutive points
    for i in range(n):
        if np.allclose(points[i], points[(i+1)%n]):
            issues.append(f"  - Duplicate consecutive points at index {i} and {i+1}")
    
    # Check for self-intersections (simple check for crossing edges)
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    def segments_intersect(p1, p2, p3, p4):
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    intersections = 0
    for i in range(n):
        for j in range(i+2, n):
            # Skip adjacent edges
            if (j - i) % n == 1 or (i - j) % n == 1:
                continue
            if segments_intersect(points[i], points[(i+1)%n], points[j], points[(j+1)%n]):
                intersections += 1
    
    if intersections > 0:
        issues.append(f"  - {intersections} self-intersections detected")
    
    # Check polygon area (degenerate if too small)
    area = 0.5 * abs(sum(points[i][0] * (points[(i+1)%n][1] - points[i-1][1]) for i in range(n)))
    if area < 1e-6:
        issues.append(f"  - Very small area: {area}")
    
    # Check for complex winding
    signed_area = 0.5 * sum(points[i][0] * points[(i+1)%n][1] - points[(i+1)%n][0] * points[i][1] for i in range(n))
    winding = "CCW" if signed_area > 0 else "CW"
    
    return {
        'polygon_id': polygon_id,
        'num_vertices': n,
        'area': abs(area),
        'signed_area': signed_area,
        'winding': winding,
        'is_valid': len(issues) == 0,
        'issues': issues
    }

def main():
    get_generic_pdk().activate()
    
    # Load the processed GDS
    processed_gds = "aim_sw/gds_file/chip_design_merged_flattened.gds"
    component = gf.import_gds(processed_gds)
    
    polygons_by_layer = component.get_polygons_points(by="tuple", layers=[(1, 0)])
    target_polygons = polygons_by_layer.get((1, 0), [])
    
    print(f"Total polygons found: {len(target_polygons)}\n")
    
    # Check each polygon
    for idx, polygon in enumerate(target_polygons, 1):
        result = check_polygon_validity(polygon, idx)
        print(f"Polygon {idx}:")
        print(f"  Vertices: {result['num_vertices']}")
        print(f"  Area: {result['area']:.2f}")
        print(f"  Winding: {result['winding']}")
        if result['issues']:
            print("  ⚠️ Issues found:")
            for issue in result['issues']:
                print(issue)
        else:
            print("  ✓ Valid")
        print()
    
    # Visualize first few polygons with focus on polygon 2
    fig, axes = plt.subplots(1, min(3, len(target_polygons)), figsize=(15, 5))
    if len(target_polygons) < 3:
        axes = np.atleast_1d(axes)
    
    for idx in range(min(3, len(target_polygons))):
        ax = axes[idx]
        polygon = target_polygons[idx]
        pts = np.array(polygon)
        
        # Plot polygon
        patch = MplPolygon(pts, closed=True, edgecolor='b', facecolor='lightblue', linewidth=1, alpha=0.7)
        ax.add_patch(patch)
        
        # Plot vertices with numbers
        for i, pt in enumerate(pts):
            ax.plot(pt[0], pt[1], 'ro', markersize=3)
            ax.text(pt[0], pt[1], str(i), fontsize=6, ha='center')
        
        # Plot edges with arrows to show direction
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i+1) % len(pts)]
            ax.arrow(p1[0], p1[1], (p2[0]-p1[0])*0.8, (p2[1]-p1[1])*0.8, 
                    head_width=20, head_length=30, fc='green', ec='green', alpha=0.3)
        
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_title(f'Polygon {idx+1} ({len(pts)} vertices)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('polygon_debug.png', dpi=150)
    print("Visualization saved to polygon_debug.png")
    plt.show()

if __name__ == '__main__':
    main()
