
from itertools import permutations
import numpy as np

def separate_left_right_rail_pixels(route):
    hist = [10000,10000]                                   # x >>> horizontal , y >>> vertical
    left_rail = []
    right_rail = []
    flag_right_rail = True
    flag_left_rail  = False

    array_route = np.array(route)
    range_x = max(array_route[:,0]) -min(array_route[:,0])
    range_y = max(array_route[:,1]) -min(array_route[:,1])
    max_dis = 0.1*(range_x**2+range_y**2)**(0.5)
    for coord in route:
        distance = ((coord[0]-hist[0])**2 + (coord[1]-hist[1])**2)**(0.5)
        if coord[1] <= hist[1] and flag_right_rail:
            right_rail.append(coord)
        elif coord[1] > hist[1] and distance <= max_dis and flag_right_rail:
            right_rail.append(coord)
        elif ((coord[1] > hist[1] or abs(coord[1]-hist[1])<5 and distance > max_dis) and distance > max_dis) and flag_right_rail:
            flag_right_rail = False
            flag_left_rail  = True
            left_rail.append(coord)
        elif flag_left_rail:
            left_rail.append(coord)

        hist = coord

    right_rail = np.array(right_rail)
    left_rail = np.array(left_rail)

    return left_rail,right_rail



def fill_rowwise(rail_sparse):
    totnum_pnts = rail_sparse.shape[0]
    appended_rail_points = []

    for idx in range(totnum_pnts - 1):

        x_i   = int(rail_sparse[idx, 0])                  # x >>> horizontal , y >>> vertical
        y_i   = int(rail_sparse[idx, 1])
        x_ii  = int(rail_sparse[(idx + 1), 0])
        y_ii  = int(rail_sparse[(idx + 1), 1])

        dy_abs = int(abs(y_i - y_ii))

        if dy_abs <= 1:
            continue


        dx_abs = int(abs(x_i - x_ii))

        type_sweep = 1                                # sweep by x=slope*y + b
        if dx_abs <= 0:
            type_sweep = 0                            # sweep by just filling the same x.

        y_s, x_s = y_ii, x_ii
        y_e, x_e = y_i, x_i

        if type_sweep == 0:
            for y in range(y_s+1, y_e):
                appended_rail_points.append([x_s,y])
        else:
            dy_fl = float(y_e - y_s)
            dx_fl = float(x_e - x_s)

            slope = dx_fl / dy_fl
            val_b = float(x_s) - slope * float(y_s)
            for y in range(y_s+1, y_e):
                x_this = slope * float(y) + val_b
                appended_rail_points.append([x_this,y])

    rail_sparse = rail_sparse.tolist()

    for item in appended_rail_points:
        rail_sparse.append(item)

    rail_dense = sorted(rail_sparse, key=lambda rail_sparse: rail_sparse[1], reverse=True)

    rail_dense = np.array(rail_dense,dtype=np.uint32)
    rail_dense = rail_dense.tolist()

    return rail_dense


def find_switches(triplets_this_image):
    totnum_routes = len(triplets_this_image)

    appended_routes_indices_t = []
    appended_routes_indices_m = []
    for i in range(totnum_routes):
        for j in range(i + 1, totnum_routes):

            route_i = triplets_this_image[i]
            route_j = triplets_this_image[j]


            ### DEALING WITH TURNOUTS FIRST ###
            x_start_i = 0.5*(route_i[0][0][0] + route_i[1][0][0])
            y_start_i = int(0.5*(route_i[0][0][1] + route_i[1][0][1]))

            x_start_j = 0.5*(route_j[0][0][0] + route_j[1][0][0])
            y_start_j = int(0.5*(route_j[0][0][1] + route_j[1][0][1]))

            # check if i is a turnout for j
            index_left  = [q for q, pixel in enumerate(route_j[1]) if pixel[1] == y_start_i if q>30]
            index_right = [q for q, pixel in enumerate(route_j[0]) if pixel[1] == y_start_i if q>30]

            if index_left and index_right:
                x_j_right = route_j[0][index_right[0]][0]
                x_j_left  = route_j[1][index_left[0]][0]

                if x_start_i <= x_j_right and x_start_i >= x_j_left:
                    appended_routes_indices_t.append([i,j])

            # check if j is a turnout for i
            index_left  = [q for q, pixel in enumerate(route_i[1]) if pixel[1] == y_start_j and q>30]
            index_right = [q for q, pixel in enumerate(route_i[0]) if pixel[1] == y_start_j and q>30]

            if index_left and index_right:
                x_i_right = route_i[0][index_right[0]][0]
                x_i_left  = route_i[1][index_left[0]][0]

                if x_start_j <= x_i_right and x_start_j >= x_i_left:
                    appended_routes_indices_t.append([j, i])

            ### DEALING WITH MERGE-INS NOW ###
            x_end_i = 0.5*(route_i[0][-1][0] + route_i[1][-1][0])
            y_end_i = int(0.5*(route_i[0][-1][1] + route_i[1][-1][1]))

            x_end_j = 0.5*(route_j[0][-1][0] + route_j[1][-1][0])
            y_end_j = int(0.5*(route_j[0][-1][1] + route_j[1][-1][1]))

            # check if i is a merge-in for j
            index_left  = [q for q, pixel in enumerate(route_j[1]) if pixel[1] == y_end_i]
            index_right = [q for q, pixel in enumerate(route_j[0]) if pixel[1] == y_end_i]


            if index_left and index_right:
                x_j_right = route_j[0][index_right[0]][0]
                x_j_left  = route_j[1][index_left[0]][0]

                if x_end_i <= x_j_right and x_end_i >= x_j_left:
                    appended_routes_indices_m.append([i,j])

            # check if j is a merge-in for i
            index_left  = [q for q, pixel in enumerate(route_i[1]) if pixel[1] == y_end_j]
            index_right = [q for q, pixel in enumerate(route_i[0]) if pixel[1] == y_end_j]

            if index_left and index_right:
                x_i_right = route_i[0][index_right[0]][0]
                x_i_left  = route_i[1][index_left[0]][0]

                if x_end_j <= x_i_right and x_end_j >= x_i_left:
                    appended_routes_indices_m.append([j,i])

    return appended_routes_indices_t, appended_routes_indices_m

def create_nodes_for_RPG(triplets_this_image, pairs_turn_out, pairs_merge_in):

    def give_start_coord_route(route):
        x = 0.5*(route[0][0][0] + route[1][0][0])
        y = 0.5*(route[0][0][1] + route[1][0][1])

        return x,y

    def give_end_coord_route(route):
        x = 0.5 * (route[0][-1][0] + route[1][-1][0])
        y = 0.5 * (route[0][-1][1] + route[1][-1][1])

        return x, y
    
    def remove_reversed_pairs(pairs):
        unique_pairs = []
        seen_pairs = set()

        for pair in pairs:
            x, y = pair
            if (y, x) not in seen_pairs:
                unique_pairs.append([x, y])
                seen_pairs.add((x, y))
        return unique_pairs

    pairs_turn_out = remove_reversed_pairs(pairs_turn_out)
    pairs_merge_in = remove_reversed_pairs(pairs_merge_in)

    turn_outs = [element[0] for element in pairs_turn_out]
    merge_ins = [element[0] for element in pairs_merge_in]

    totnum_routes      = len(triplets_this_image)
    RPG                = []
    main_routes        = []
    max_len_each_route = 6

    if max_len_each_route > totnum_routes:
        max_len_each_route = totnum_routes

    for len_route in range(1,max_len_each_route+1):
        permutations_list = list(permutations(range(0, totnum_routes), len_route))

        for permutation in permutations_list:
            if len_route == 1:
                if permutation[0] not in turn_outs and permutation[0] not in merge_ins:
                    RPG.append([permutation[0]])
                    main_routes.append(permutation[0])

            else:
                consecutiveness = 0
                y_sequence = []
                for counter in range(0,len_route-1):
                    this_route_idx = permutation[counter]
                    next_route_idx = permutation[counter+1]
                    x_this_route_start, y_this_route_start = give_start_coord_route(triplets_this_image[this_route_idx])
                    x_this_route_end, y_this_route_end     = give_end_coord_route(triplets_this_image[this_route_idx])
                    x_next_route_start, y_next_route_start = give_start_coord_route(triplets_this_image[next_route_idx])
                    x_next_route_end, y_next_route_end     = give_end_coord_route(triplets_this_image[next_route_idx])

                    if [next_route_idx,this_route_idx] in pairs_turn_out or [this_route_idx,next_route_idx] in pairs_merge_in:
                        consecutiveness += 1


                    if [next_route_idx, this_route_idx] in pairs_turn_out:
                        y_sequence.append(y_next_route_start)
                    elif [this_route_idx,next_route_idx] in pairs_merge_in:
                        y_sequence.append(y_this_route_end)





                condition_1 = consecutiveness == len_route - 1                                                                    # back-to-back routes
                condition_2 = True
                if len(y_sequence) > 1:                                                                                           # y of the starting point of turnouts should go down
                    for k in range(len(y_sequence)-1):
                        if y_sequence[k+1] > y_sequence[k]:
                            condition_2 = False
                            break

                condition_3 = False                                                                                               # a route ends with either a main route or isolated turnout
                if permutation[-1] in main_routes or (permutation[-1] in turn_outs and permutation[-1] not in merge_ins):
                    condition_3 = True

                condition_4 = False
                if permutation[0] in main_routes or (permutation[0] in merge_ins and permutation[0] not in turn_outs):
                    condition_4 = True


                if condition_1 and condition_2 and condition_3 and condition_4:
                    list_path_this = []
                    for item in permutation:
                        list_path_this.append(item)
                    RPG.append(list_path_this)


    return RPG


def create_unified_routes(all_routes, RPG, turnout_pairs, mergein_pairs):
    all_unified_routes       = []

    for unified_route_indices in RPG:
        this_unfied_route = [[], []]
        num_routes_this_unified_route = len(unified_route_indices)
        for RL_idx in range(0,2):
            for q in range(num_routes_this_unified_route):

                if num_routes_this_unified_route == 1:
                    for pixel in all_routes[unified_route_indices[q]][RL_idx]:
                        this_unfied_route[RL_idx].append(pixel)
                elif num_routes_this_unified_route > 1:

                    if q == 0:
                        for pixel in all_routes[unified_route_indices[q]][RL_idx]:
                            if (pixel[1] > all_routes[unified_route_indices[q+1]][RL_idx][0][1] and [unified_route_indices[q+1],unified_route_indices[q]] in turnout_pairs) or \
                                    [unified_route_indices[q],unified_route_indices[q+1]] in mergein_pairs:
                                this_unfied_route[RL_idx].append(pixel)
                    elif q == num_routes_this_unified_route - 1:
                        for pixel in all_routes[unified_route_indices[q]][RL_idx]:
                            if (pixel[1] < all_routes[unified_route_indices[q-1]][RL_idx][-1][1] and [unified_route_indices[q-1],unified_route_indices[q]] in mergein_pairs) or \
                                [unified_route_indices[q],unified_route_indices[q-1]] in turnout_pairs:
                                this_unfied_route[RL_idx].append(pixel)

                    else:
                        for pixel in all_routes[unified_route_indices[q]][RL_idx]:
                            if [unified_route_indices[q], unified_route_indices[q-1]] in turnout_pairs and [unified_route_indices[q], unified_route_indices[q+1]] in mergein_pairs:
                                this_unfied_route[RL_idx].append(pixel)
                            elif [unified_route_indices[q + 1], unified_route_indices[q]] in turnout_pairs and [unified_route_indices[q], unified_route_indices[q-1]] in turnout_pairs and \
                                    pixel[1] > all_routes[unified_route_indices[q+1]][RL_idx][0][1]:
                                this_unfied_route[RL_idx].append(pixel)
                            elif [unified_route_indices[q], unified_route_indices[q+1]] in mergein_pairs and [unified_route_indices[q-1], unified_route_indices[q]] in mergein_pairs and \
                                    pixel[1] < all_routes[unified_route_indices[q-1]][RL_idx][-1][1]:
                                this_unfied_route[RL_idx].append(pixel)
                            else:
                                if pixel[1] < all_routes[unified_route_indices[q-1]][RL_idx][-1][1] and pixel[1] > all_routes[unified_route_indices[q+1]][RL_idx][0][1]:
                                    this_unfied_route[RL_idx].append(pixel)

        all_unified_routes.append(this_unfied_route)

    return all_unified_routes


def path_unifier(labels_raw):
    min_len_route       = 2
    triplets_this_image = []
    data_json_this      = labels_raw
    for route in data_json_this:

        triplets_this_route = []
        if len(route) < min_len_route:
            continue


        left_rail_sparse, right_rail_sparse = separate_left_right_rail_pixels(route)
        if len(left_rail_sparse) == 0 or len(right_rail_sparse) == 0:
            continue

        left_rail_dense   = fill_rowwise(left_rail_sparse)
        right_rail_dense  = fill_rowwise(right_rail_sparse)


        triplets_this_route.append(right_rail_dense)
        triplets_this_route.append(left_rail_dense)

        triplets_this_image.append(triplets_this_route)


    appended_routes_indices_t, appended_routes_indices_m = find_switches(triplets_this_image)
    rail_path_graph    = create_nodes_for_RPG(triplets_this_image, appended_routes_indices_t, appended_routes_indices_m)
    rail_path_graph    = sorted(rail_path_graph, key=lambda x: x[0])
    rail_path_graph    = [vec for vec in rail_path_graph if len(vec) == 1 or vec[0] != vec[-1]]
    all_unified_routes = create_unified_routes(triplets_this_image, rail_path_graph, appended_routes_indices_t, appended_routes_indices_m)

    return all_unified_routes
