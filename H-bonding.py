import argparse
import MDAnalysis as mda
import re
import numpy as np
from scipy.spatial import cKDTree

def get_coordinates(topol_path,trr_path,start_frame,end_frame):
    #initialize the universe object
    u = mda.Universe(topol_path, trr_path)

    #get the number of atoms per molecule
    start=u.select_atoms('name C1P')[0]
    end = u.select_atoms('name C1P')[1]
    num_atom_per_molecule = end.ix - start.ix

    #get the number of molecules per system
    all_atom=u.atoms
    num_mole_per_sys=int(len(all_atom)/num_atom_per_molecule)

    #get the atom group of oxygen and hydrogen on the hydroxyl group, changes per frame
    oxygen=u.select_atoms('name O*')
    hydrogen=u.select_atoms('name HO*')

    #get the time step of the simulation
    t1=u.trajectory[0].time
    t2=u.trajectory[1].time
    step=t2-t1 #unit: ps

    #get the start step and end step
    s1=int(start_frame//step)
    s2=int(end_frame//step)

    # Function to extract the trailing number from an atom name
    def extract_number(atom_name):
        match = re.search(r'\d+', atom_name)
        return int(match.group()) if match else None

    #initialize the dictionary to store the coordinates, key is the frame, value is the list of list of coordinates per molecule
    coordinates={}

    #initialize the dictionary to store the dimensions, key is the frame, value is the list of box dimensions for each frame
    dimensions={}

    #iterate over the frames of interest
    for i in range(s1,s2 + 1):
        # get the dimension of the box; note that the actual dimension changes with frame
        u.trajectory[i]
        dimension = u.dimensions[:3]

        frame_coords = [[] for _ in range(num_mole_per_sys)]
        coordinates[int(i*2000)] = frame_coords
        dimensions[int(i*2000)]=dimension

        # Iterate over all oxygen atoms
        for o_atom in oxygen:
            o_num = extract_number(o_atom.name)
            o_resname = o_atom.residue.resname
            o_index=o_atom.ix//num_atom_per_molecule

            # Find the matching hydrogen atom
            matched_h = None
            for h_atom in hydrogen:
                h_num=extract_number(h_atom.name)
                h_resname = o_atom.residue.resname
                h_index=h_atom.ix//num_atom_per_molecule
                if o_num==h_num and o_resname==h_resname and o_index==h_index:
                    matched_h = h_atom.position
                    break

            # Store coordinates in the respective molecule list
            if matched_h is not None:
                frame_coords[o_index].append([o_atom.position, matched_h])
            else:
                frame_coords[o_index].append([o_atom.position])
    return coordinates,dimensions


def vector_PBC(vec,box_dim):
    return vec-np.round(vec/box_dim)*box_dim

def angle_check_PBC(o_donate,h,o_accept,box_dim,constraint_a):
    o_donate=np.array(o_donate)
    h=np.array(h)
    o_accept=np.array(o_accept)

    v_h_od=vector_PBC(h-o_donate,box_dim)
    v_h_oa=vector_PBC(h-o_accept,box_dim)

    dot = np.dot(v_h_od, v_h_oa)
    v1n = np.linalg.norm(v_h_od)
    v2n = np.linalg.norm(v_h_oa)
    angle = 180 * (np.arccos(dot / (v1n * v2n)) / np.pi)
    return angle >= constraint_a

def distance_check_PBC(o1,o2,box_dim,constraint_d):
    o1=np.array(o1)
    o2=np.array(o2)
    delta=o1-o2
    delta-=np.round(delta / box_dim) * box_dim
    d = np.linalg.norm(delta)
    return d<=constraint_d

def compute_intra_hbonds(constraint_a, constraint_d,box_dim,frame):
    total_hbonds=0
    total_molecule=len(frame)
    for molecule in frame:
        for i, group_i in enumerate(molecule):
            if len(group_i)==1:
                continue
            else:
                for j, group_j in enumerate(molecule):
                    if i==j:
                        continue
                    if distance_check_PBC(group_i[0],group_j[0],box_dim,constraint_d) and\
                        angle_check_PBC(group_i[0],group_i[1],group_j[0],box_dim,constraint_a):
                        total_hbonds+=1
    avg_hbonds=round(total_hbonds/total_molecule,4)
    return avg_hbonds

def mirror_box(box_dim,frame):
    frame=np.array(frame)
    mirrored_coords = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                shift = np.array([x * box_dim[0], y * box_dim[1], z * box_dim[2]])
                mirrored_coords.extend(frame + shift)
    return np.array(mirrored_coords)

def compute_total_hbonds(constraint_a, constraint_d,dimension, frame):
    #initialize the total hydrogen bonding
    total_hbonds=0

    #extract oxygen coordinates coordinates
    oxygen_coords=[]
    for molecule in frame:
        oxygen_coords += [group[0] for group in molecule]

    #duplicate the box to consider PBC condition
    mirrored_oxygen_coords=mirror_box(dimension,oxygen_coords)
    # Create KDTree for oxygen
    oxygen_tree = cKDTree(mirrored_oxygen_coords)

    # Iterate over hydroxyl groups to find potential hydrogen bonds
    for molecule in frame:
        for i, group in enumerate(molecule):
            if len(group) == 2:  # Only consider hydroxyl groups
                O_coord = group[0]
                H_coord = group[1]

                test=np.array(H_coord)
                truth=np.array(([27.641663, 28.999205, 12.95134 ]))
                if np.array_equal(test,truth):
                    print('frame match')

                # Find nearby oxygen atoms within cutoff
                nearby_oxygens = oxygen_tree.query_ball_point(O_coord, constraint_d)
                for j in nearby_oxygens:
                    target_O_coord = mirrored_oxygen_coords[j]
                    # Check for hydrogen bonding criteria with target_O_coord
                    if np.array_equal(target_O_coord,O_coord):
                        continue
                    elif angle_check_PBC(O_coord,H_coord,target_O_coord,dimension,constraint_a):
                        total_hbonds+=1
    avg_total_hbonds=round(total_hbonds/len(frame),4)
    return avg_total_hbonds

def main(topol_path,trr_path,constraint_a, constraint_d,start_frame,end_frame):
    coordinates,dimensions=get_coordinates(topol_path, trr_path, start_frame, end_frame)
    intra_hbonds_count=0
    total_hbonds_count=0

    #iterate over all the frames to do the averaging
    for frame in coordinates:
        intra_hbonds_per_molecule=compute_intra_hbonds(constraint_a, constraint_d, dimensions[frame],coordinates[frame])
        total_hbonds_per_molecule=compute_total_hbonds(constraint_a, constraint_d, dimensions[frame], coordinates[frame])
        intra_hbonds_count+=intra_hbonds_per_molecule
        total_hbonds_count+=total_hbonds_per_molecule

    #h-bonding per molecule
    intra_hbonds=intra_hbonds_count/len(coordinates)
    total_hbonds=total_hbonds_count/len(coordinates)
    inter_hbonds=total_hbonds-intra_hbonds
    return intra_hbonds, inter_hbonds, total_hbonds

if __name__=='__main__':
    # initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--trr_path',
                        type=str,
                        default='./mda_test/traj.trr',
                        help="path for the trr file")
    parser.add_argument('--topol_path',
                        type=str,
                        default='./mda_test/topol.tpr',
                        help="path for the topol file")
    parser.add_argument('--start_frame',
                        type=int,
                        default=0,
                        help="the first frame to do the calculation, unit: ps")
    parser.add_argument('--end_frame',
                        type=int,
                        default=100000,
                        help="the last frame to do the calculation, unit: ps")
    parser.add_argument('--constraint_a',
                        type=int,
                        default=130,
                        help="O-H-O angle constraint for the H-bonding, unit: degree")
    parser.add_argument('--constraint_d',
                        type=int,
                        default=3.5,
                        help="O-O distance constraint for the H-bonding, unit: A")

    # Parse the argument
    args = parser.parse_args()
    intra_hbonds, inter_hbonds, total_hbonds=main(args.topol_path,args.trr_path,args.constraint_a, args.constraint_d,args.start_frame,args.end_frame)
    print(f'averaged number of H-bonds over frames from {args.start_frame} to {args.end_frame} \n intra_molecular: {round(intra_hbonds,2)} \n inter_molecular: {round(inter_hbonds,2)} \n total number of hbonds: {round(total_hbonds,2)}')