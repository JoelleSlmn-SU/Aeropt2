import os, sys

sys.path.append(os.path.dirname("FileRW"))
sys.path.append(os.path.dirname("Utilities"))
from Utilities.PointClouds import dist_between_points
from FileRW.Mesh import Mesh

def average_node_spacing(ff : Mesh, ignore=[1,2,3,4]):
    # 1. Node Spacing - Average spacing of surrounding edges. 
    #   - loop through each triangle and add edge distance to dictionarys
    #   - calculate average 
    #   - plot
    tot_spacing = {}
    counted     = {}
    edge_count  = {}
    

    for i in range(ff.node_count):
        tot_spacing[i] = 0.0
        edge_count[i]  = 0.0
        counted[i]     = []

    for f3 in ff.boundary_triangles:
        if f3[3] in ignore:
            continue
        v1 = f3[0]
        v2 = f3[1]
        v3 = f3[2]

        p1 = ff.nodes[v1][0:3]
        p2 = ff.nodes[v2][0:3]
        p3 = ff.nodes[v3][0:3]
        #P1
        if v2 not in counted[v1]:
            tot_spacing[v1] += dist_between_points(p1, p2)
            edge_count[v1] += 1.0
            counted[v1].append(v2)
        if v3 not in counted[v1]:
            tot_spacing[v1] += dist_between_points(p1, p3)
            edge_count[v1] += 1.0
            counted[v1].append(v3)
        #P2
        if v1 not in counted[v2]:
            tot_spacing[v2] += dist_between_points(p2, p1)
            edge_count[v2] += 1.0
            counted[v2].append(v1)
        if v3 not in counted[v2]:
            tot_spacing[v2] += dist_between_points(p2, p3)
            edge_count[v2] += 1.0
            counted[v2].append(v3)
        #P3
        if v2 not in counted[v3]:
            tot_spacing[v3] += dist_between_points(p3, p2)
            edge_count[v3] += 1.0
            counted[v3].append(v2)
        if v1 not in counted[v3]:
            tot_spacing[v3] += dist_between_points(p3, p1)
            edge_count[v3] += 1.0
            counted[v3].append(v1)

    for f4 in ff.boundary_quads:
        if f4[4] in ignore:
            continue
        v1 = f4[0]
        v2 = f4[1]
        v3 = f4[2]
        v4 = f4[3]

        p1 = ff.nodes[v1][0:3]
        p2 = ff.nodes[v2][0:3]
        p3 = ff.nodes[v3][0:3]
        p4 = ff.nodes[v4][0:3]
        #P1
        if v2 not in counted[v1]:
            tot_spacing[v1] += dist_between_points(p1, p2)
            edge_count[v1] += 1.0
            counted[v1].append(v2)
        if v3 not in counted[v1]:
            tot_spacing[v1] += dist_between_points(p1, p3)
            edge_count[v1] += 1.0
            counted[v1].append(v3)
        if v4 not in counted[v1]:
            tot_spacing[v1] += dist_between_points(p1, p4)
            edge_count[v1] += 1.0
            counted[v1].append(v4)
        #P2
        if v1 not in counted[v2]:
            tot_spacing[v2] += dist_between_points(p2, p1)
            edge_count[v2] += 1.0
            counted[v2].append(v1)
        if v3 not in counted[v2]:
            tot_spacing[v2] += dist_between_points(p2, p3)
            edge_count[v2] += 1.0
            counted[v2].append(v3)
        if v4 not in counted[v2]:
            tot_spacing[v2] += dist_between_points(p2, p4)
            edge_count[v2] += 1.0
            counted[v2].append(v4)
        #P3
        if v2 not in counted[v3]:
            tot_spacing[v3] += dist_between_points(p3, p2)
            edge_count[v3] += 1.0
            counted[v3].append(v2)
        if v1 not in counted[v3]:
            tot_spacing[v3] += dist_between_points(p3, p1)
            edge_count[v3] += 1.0
            counted[v3].append(v1)
        if v4 not in counted[v3]:
            tot_spacing[v3] += dist_between_points(p3, p4)
            edge_count[v3] += 1.0
            counted[v3].append(v4)
        #P4
        if v1 not in counted[v4]:
            tot_spacing[v4] += dist_between_points(p4, p1)
            edge_count[v4] += 1.0
            counted[v4].append(v1)
        if v2 not in counted[v4]:
            tot_spacing[v4] += dist_between_points(p4, p2)
            edge_count[v4] += 1.0
            counted[v4].append(v2)
        if v3 not in counted[v4]:
            tot_spacing[v4] += dist_between_points(p4, p3)
            edge_count[v4] += 1.0
            counted[v4].append(v3)

    avg_spacing = []
    
    for i in range(ff.node_count):
        if edge_count[i] == 0.0:
            continue
        avg_spacing.append(tot_spacing[i]/edge_count[i])
        
    return avg_spacing
