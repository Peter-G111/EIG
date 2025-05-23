# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Julia 1.10
#     language: julia
#     name: julia-1.10
# ---

# ### Aim
# A notebook that solves Escape Interdiction Games (EIG) according to the methods described in the original paper by Zhang et al., as well as using the methods in the intermediate report.
# Both processes rely on an LP **CoreLP** which finds an optimal mixed strategy given a (small) subset of available strategies for both players.
# After finding optimal mixed strategies, we use defender and attacker oracles to find a response to the most recent optimal mixed strategy of the opposition, then add the support of the response to the set of available strategies.
# We iterate until no new improving strategies can be found in consulting the oracles.

# ### Data & Generating EIG Instances
# An EIG requires a directed network structure to model a city's road network.
# We represent networks as a list of directed edges.
# A directed edge from node `i` to `j` is given by `(i, j, d)` where `i` and `j` are nodes and `d` is the distance from `i` to `j`.
# In this implementation we use a grid graph, and represent a node with a pair `(a, b)` corresponding to the row and column of the node in the grid graph.
#
# An attacker strategy is a list `att` of 2-tuples `(v_i, t_i)` where the `i`th state entails `t_i` as the arrival time at node `v_i`.
# It is also a named tuple, with labels `v` and `t` respectively.
#
# A defender strategy is a list `def` of lists `def[r]` of 3-tuples `(v_j^r, t_j^r-, t_j^r+)` where the `j`th state entails `t_j^r-` as the arrival time at node `v_j^r` and `t_j^r+` as the departure time for resource `r`.
# It is also a named tuple, with labels `v`, `t_a`, `t_b` respectively.
#
# The time-discretisation parameter is `δ`, and the finite-horizon max-time is `t_max`.

# using Random
# using Distributions
using JuMP
using Gurobi
# using Plots
using DataStructures  # binary heap for Dijkstra's Algorithm, default dictionary
# using Formatting
using CSV  # for CSV reading (input)
using JSON  # for writing to JSON file (output)

# +
struct Edge
    a::Int  # start node of edge
    b::Int  # end node of edge
    dist::Float64
end

struct NetworkGraph
    num_nodes::Int
    edges::Vector{Edge}
    v_0::Int  # attacker start node
    v_0r::Vector{Int}  # defender start nodes
    exit_nodes::Vector{Int}  # nodes with an outgoing edge to v_∞
    v_∞::Int  # attacker escape node
end

struct EIG
    network::NetworkGraph
    δ
    t_max
    dist_mtx
    neighbourhoods  # list of lists of (outgoing) neighbourhoods for each vertex
    speed_A
    speed_D
    # node_int_to_pair  # converts a node int representation into a pair of int representation (with coordinates interpretation)
end

node_int_to_pair(x, width) = ((x-1) ÷ width + 1, (x - 1) % width + 1)  # converts a node int representation into a pair of int representation (with coordinates interpretation)

# node_int_to_pair = ((x-1) ÷ WIDTH + 1, (x - 1) % WIDTH + 1)  # converts a node int representation into a pair of int representation (with coordinates interpretation)

# +
function generatenetwork(;length=3, width=3, orth_edge_prob=0.4, diag_edge_prob=0.2, num_defenders=4, num_exit_nodes=3, rounding=false)
    """
    Returns a grid (square) network where each pair of orthogonal (so vertical or horizontal) vertices has an edge with certain probabilities,
    and designates the attacker start node and defender resource start nodes.
    """
    # num_nodes = length * width;
    nodes = [(i, j) for i=1:length for j=1:width];  # list of nodes, pairs of integers
    # nodes = [i for i in 1:length * width];  # list of nodes, pairs of integers
    
    # determine edges
    edges = []
    
    # vertical edges
    for i=1:length-1 for j=1:width
            if rand() < orth_edge_prob; push!(edges, ((i, j), (i+1, j), generatedist(rounding=rounding))); end
            if rand() < orth_edge_prob; push!(edges, ((i+1, j), (i, j), generatedist(rounding=rounding))); end  # other direction (independent of first direction)
    end end
    
    # horizontal edges
    for i=1:length for j=1:width-1
            if rand() < orth_edge_prob; push!(edges, ((i, j), (i, j+1), generatedist(rounding=rounding))); end
            if rand() < orth_edge_prob; push!(edges, ((i, j+1), (i, j), generatedist(rounding=rounding))); end
    end end
    
    # N-diagonal edges
    for i=1:length-1 for j=1:width-1
            if rand() < diag_edge_prob; push!(edges, ((i, j), (i+1, j+1), generatedist(rounding=rounding))); end
            if rand() < diag_edge_prob; push!(edges, ((i+1, j+1), (i, j), generatedist(rounding=rounding))); end
    end end
    
    # Z-diagonal edges
    for i=1:length-1 for j=1:width-1
            if rand() < diag_edge_prob; push!(edges, ((i+1, j), (i, j+1), generatedist(rounding=rounding))); end
            if rand() < diag_edge_prob; push!(edges, ((i, j+1), (i+1, j), generatedist(rounding=rounding))); end
    end end
    
    # add v_∞ node adjacent to all nodes on the border of the grid
    v_infty_node = (length, width+1)  # represented as this so that the integer form is 1 above the greatest internal node
    push!(nodes, v_infty_node)
    
    push!(edges, (v_infty_node, v_infty_node, 0.0))  # add loop at v_∞ node
    
    # choose exit nodes on border (implicitly connected to the ultimate exit node v_∞); first create list of border nodes
    border_nodes = []
    for i=1:length
        push!(border_nodes, (i, 1));
        push!(border_nodes, (i, width));
    end
    for j=2:width-1
        push!(border_nodes, (1, j));
        push!(border_nodes, (length, j));
    end
    
    exit_nodes = sample(border_nodes, num_exit_nodes, replace=false)
    
    for node in exit_nodes
        push!(edges, (node, v_infty_node, generatedist(rounding=rounding)));
    end
    
#     # top and bottom border nodes
#     for i=1:length
#         push!(edges, ((i, 1), v_infty_node, generatedist()));
#         push!(edges, ((i, width), v_infty_node, generatedist()));
#     end
    
#     # left and right border nodes
#     for j=2:width-1
#         push!(edges, ((1, j), v_infty_node, generatedist()));
#         push!(edges, ((length, j), v_infty_node, generatedist()));
#     end
    
    # pick attacker start node in middle of grid
    att_start_node = ((length + 1) ÷ 2, (width + 1) ÷ 2)
    
    # pick defender resource start nodes
    def_start_nodes = []
    for resource=1:num_defenders
        # generate random node, but ensure it is not the attacker start node
        new_node = (rand(1:length), rand(1:width))
        while new_node == att_start_node
            new_node = (rand(1:length), rand(1:width))
        end
        push!(def_start_nodes, new_node)
    end
    
    # sort edges by first vertex
    sort!(edges)
    
    new_nodes, new_edges, new_att_start_node, new_def_start_nodes, new_exit_nodes = convertEIGnodestointegerform(nodes, edges, att_start_node, def_start_nodes, exit_nodes, width=width)
    
    # display(nodes)
    # display(new_edges)
    
    edges_ = [Edge(e[1], e[2], e[3]) for e in new_edges]
    
    network = NetworkGraph(size(new_nodes)[1], edges_, new_att_start_node, new_def_start_nodes, new_exit_nodes, size(new_nodes)[1])
    return network
    # return (nodes, edges, att_start_node, def_start_nodes, exit_nodes)
end
# -

function generatedist(;C_e=6, rounding=false)
    """
    Returns a travel time according to eq (3) in Zhang et al.
    Note if T_e, f_e and C_e are rational, then the generated distance is rational. 
    """
    T_e = rand(1:10)
    f_e = rand(1:6)
    
    result = T_e * (1 + 0.15 * (f_e / C_e)^4)

    return rounding ? round(result) : result
end

function convertEIGnodestointegerform(nodes, edges, att_start_node, def_start_nodes, exit_nodes; width)
    """
    Helper function for generating EIG networks.
    """
    mapping(x) = width*(x[1]-1) + x[2]
    
    new_nodes = 1:length(nodes)
    
    new_edges = []
    for edge in edges
        push!(new_edges, (mapping(edge[1]), mapping(edge[2]), edge[3]))
    end
    
    new_att_start_node = mapping(att_start_node)
    
    new_def_start_nodes = []
    for node in def_start_nodes
        push!(new_def_start_nodes, mapping(node))
    end
    
    new_exit_nodes = []
    for node in exit_nodes
        push!(new_exit_nodes, mapping(node))
    end
    
    return new_nodes, new_edges, new_att_start_node, new_def_start_nodes, new_exit_nodes
end

function computedistancematrix(network::NetworkGraph)
    """
    Returns a matrix whose ijth entry is the distance from node i to node j
    Employs Floyd–Warshall algorithm (all pairs distances)
    """
    n = network.num_nodes
    dist = fill(Inf, (n, n))
    
    for e in network.edges
        dist[e.a, e.b] = e.dist
    end
    for i=1:n
        dist[i, i] = 0
    end
    for k=1:n
        for i=1:n
            for j=1:n
                potential_dist = dist[i, k] + dist[k, j]
                if dist[i, j] > potential_dist
                    dist[i, j] = potential_dist
                end
            end
        end
    end
    
    return dist
end

# +
function computeneighbourhoods(network::NetworkGraph)
    """
    Returns, for each node v, a list of vertices [u] such that vu is an edge/arc 
    (Assumes integer nodes)
    """
    neighbourhoods = [[] for _ in 1:network.num_nodes]  # all empty lists to begin
    for e in network.edges
        # update appropriate neighbourhood
        push!(neighbourhoods[e.a], e.b)
    end
    
    # push!(neighbourhoods[network.v_∞], network.v_∞)  # add loop at v_∞ node (to allow attacker strategies to end on v_∞ node prematurely)
    
    return neighbourhoods
end

# function computeneighbourhoods(nodes, edges)
#     """
#     Returns, for each node v, a list of vertices [u] such that vu is an edge/arc 
#     (Assumes integer nodes)
#     """
#     neighbourhoods = [[] for _ in nodes]
#     for edge in edges
#         # update appropriate neighbourhood
#         push!(neighbourhoods[edge[1]], edge[2])
#     end
#     return neighbourhoods
# end

# +
# WIDTH = 3
# LENGTH = 3

# EIG_not_generated = true

# network = nothing
# dist_mtx = nothing

# # nodes = nothing
# # edges = nothing
# # v_0 = nothing
# # v_0r = nothing
# # exit_nodes = nothing

# # nodes_p = nothing
# # edges_p = nothing
# # att_start_node_p = nothing
# # def_start_nodes_p = nothing
# # exit_nodes_p = nothing


# while EIG_not_generated
#     # nodes_p, edges_p, att_start_node_p, def_start_nodes_p, exit_nodes_p = generatenetwork(WIDTH);

#     # nodes, edges, v_0, v_0r, exit_nodes = convertEIGnodestointegerform(nodes_p, edges_p, att_start_node_p, def_start_nodes_p, exit_nodes_p)
#     network = generatenetwork(width=WIDTH, length=LENGTH, num_defenders=3, orth_edge_prob=0.7, diag_edge_prob=0.4, rounding=true)
#     # generatenetwork(;length=3, width=3, orth_edge_prob=0.4, diag_edge_prob=0.2, num_defenders=4, num_exit_nodes=3)
#     dist_mtx = computedistancematrix(network);

#     # display(dist_mtx)
    
#     EIG_not_generated = false
    
#     if dist_mtx[network.v_0, network.num_nodes] == Inf
#         println("!!!!!!!!!!!!!!!!!No path from v_0 to v_∞")
#         EIG_not_generated = true
#     end
# end

# # chosen T_MAX and DELTA values (paper doesn't have explicit choices)
# DELTA = 1
# T_MAX = maximum(filter(!isinf, dist_mtx))

# neighbourhoods = computeneighbourhoods(network);

# node_int_to_pair(x) = ((x-1) ÷ WIDTH + 1, (x - 1) % WIDTH + 1)  # converts a node int representation into a pair of int representation (with coordinates interpretation)

# the_eig = EIG(network, DELTA, T_MAX, dist_mtx, neighbourhoods, node_int_to_pair)
# -

# ### Solving

# #### Heuristic initial strategies

function shortest_path(eig::EIG, init_node::Int, destination::Int; scale_divisor=1)
    """
    Returns a shortest path of the form [(v_i, t_i) for i]
    Uses Dijkstra's algorithm

    scale_divisor: Divides times (t_i) by scale_divisor at the very end once the shortest path is found.
    """
    # find path from v_0 to v_∞
    
    max_dist = maximum(filter(!isinf, eig.dist_mtx))
    
    dist = [Inf for _ in 1:eig.network.num_nodes]
    prev = [-1 for _ in 1:eig.network.num_nodes]
    
    dist[init_node] = 0
    
    # add all vertices to min-heap
    minheap = MutableBinaryMinHeap{Float64}()
    # minheap = BinaryMinHeap([])
    
    for node in 1:eig.network.num_nodes
        # push!(minheap, [dist[node], node])  # ordered by distance (first entry in tuple)
        node_heap_id = push!(minheap, dist[node])  # save distance into heap; node is encoded in the handle 
    end
    
    # vertexlist = [node for node in nodes]  # quick implementation
    
    while length(minheap) > 0  # heap not empty
        # find vertex still in vertexlist with least distance from v_0
        # d_u, u = pop!(minheap)
        
        # d_u = pop!(minheap)
        d_u, u = top_with_handle(minheap)
        pop!(minheap)
        
        # display((d_u, u))
        
        # u = argmin(dist)  # 
        # deleteat!(vertexlist, u)
        # u = Int(u)  # convert node to integer (as it is a float when in the heap)
        # display(u)
        
        for v in eig.neighbourhoods[u]
            altdist = dist[u] + eig.dist_mtx[u, v]
            if altdist < dist[v]
                dist[v] = altdist
                
                update!(minheap, v, altdist)  # also update heap 
                
                prev[v] = u
            end
        end
    end
    
    # display(prev)
    
    # construct sequence of nodes to traverse (as well as distances)
    seq_rev = []
    # u = length(nodes)  # begin at v_∞
    
    u = destination
    
    # push!(seq_rev, (v=u, t=dist[destination]))
    # u = prev[u]

    while prev[u] != -1
        push!(seq_rev, (v=u, t=dist[u] / scale_divisor))  # divide by scale_divisor
        u = prev[u]
    end

    push!(seq_rev, (v=init_node, t=0.0))
    
    return reverse(seq_rev)
end

# +
function makeheuristicattacker(eig::EIG)
    """
    Generates a simple attacker strategy where the attacker heads towards an exit node as quickly as possible.
    Uses Dijkstra's algorithm
    """
    
    path = shortest_path(eig, eig.network.v_0, eig.network.v_∞, scale_divisor=eig.speed_A)
    path[end] = (v=eig.network.v_∞, t=eig.t_max)  # ensure escape at t_max 
    return path
    
    
#     # find path from v_0 to v_∞
    
#     max_dist = maximum(filter(!isinf, eig.dist_mtx))
    
#     dist = [max_dist for _ in 1:eig.network.num_nodes]
#     prev = [-1 for _ in 1:eig.network.num_nodes]
    
#     dist[eig.network.v_0] = 0
    
#     # add all vertices to min-heap
#     minheap = MutableBinaryMinHeap{Float64}()
#     # minheap = BinaryMinHeap([])
    
#     for node in 1:eig.network.num_nodes
#         # push!(minheap, [dist[node], node])  # ordered by distance (first entry in tuple)
#         node_heap_id = push!(minheap, dist[node])  # save distance into heap; node is encoded in the handle 
#     end
    
#     # vertexlist = [node for node in nodes]  # quick implementation
    
#     while length(minheap) > 0  # heap not empty
#         # find vertex still in vertexlist with least distance from v_0
#         # d_u, u = pop!(minheap)
        
#         # d_u = pop!(minheap)
#         d_u, u = top_with_handle(minheap)
#         pop!(minheap)
        
#         # display((d_u, u))
        
#         # u = argmin(dist)  # 
#         # deleteat!(vertexlist, u)
#         # u = Int(u)  # convert node to integer (as it is a float when in the heap)
#         # display(u)
        
#         for v in eig.neighbourhoods[u]
#             altdist = dist[u] + eig.dist_mtx[u, v]
#             if altdist < dist[v]
#                 dist[v] = altdist
                
#                 update!(minheap, v, altdist)  # also update heap 
                
#                 prev[v] = u
#             end
#         end
#     end
    
#     # display(prev)
    
#     # construct sequence of nodes to traverse (as well as distances)
#     seq_rev = []
#     # u = length(nodes)  # begin at v_∞
    
#     u = eig.network.v_∞
    
#     push!(seq_rev, (v=u, t=eig.t_max))  # ensure escape at t_max
#     u = prev[u]

#     while prev[u] != -1
#         push!(seq_rev, (v=u, t=dist[u]))
#         u = prev[u]
#     end

#     push!(seq_rev, (v=eig.network.v_0, t=0.0))
    
#     return reverse(seq_rev)
end
# -

function makeheuristicdefender(eig::EIG)
    """
    Generates a simple defender strategy where the defender stays at their start locations.
    """
    return [[(v=init_node, t_a=0.0, t_b=ceil(eig.t_max))] for init_node in eig.network.v_0r]
end

function makedefres_path(eig::EIG, resource_idx, end_node)
    """
    Returns a defender path where resource goes straight to end_node (via a shortest path), then waits there forever.
    If the resource cannot reach end_node within t_max, then the resource just stays at their v_0r forever.
    """
    if end_node == eig.network.v_0r[resource_idx]
        return [(v=end_node, t_a = 0.0, t_b=ceil(eig.t_max))]  # stay on exit node forever
    end
    
    if div(eig.dist_mtx[eig.network.v_0r[resource_idx], end_node], eig.speed_D) > eig.t_max
        return [(v=eig.network.v_0r[resource_idx], t_a = 0.0, t_b=ceil(eig.t_max))]  # stay at v_0r forever
    end
    
    path = shortest_path(eig, eig.network.v_0r[resource_idx], end_node, scale_divisor=eig.speed_D)
    
    return [
        (v=eig.network.v_0r[resource_idx], t_a = 0.0, t_b=0.0),
        (v=end_node, t_a=path[end].t, t_b=ceil(eig.t_max))
    ]
end

function makedefstrat_path(eig, end_nodes)
    """
    Returns a defender strategy where resource goes straight to an end_node (via a shortest path), then waits there forever.
    If a resource cannot reach their specified end_node, then the resource just stays at their v_0r forever (this is so that we can animate).
    """
    return [makedefres_path(eig, r, end_nodes[r]) for r=1:length(eig.network.v_0r)]
end

function makedefstrat_endnodes(eig)
    """
    Returns a defender strategy where each resource goes straight to an exit node
    """
    exit_nodes = eig.network.exit_nodes
    num_res = length(eig.network.v_0r)
    
    # cycle through exit_nodes, sending resource straight to exit node
    end_nodes = [exit_nodes[idx % num_res + 1] for idx in 1:num_res]  # end nodes for each resource (can be more clever with choice, but this is just a quick code)
    
    return makedefstrat_path(eig, end_nodes)
end

# # #### Visualisation

# function drawEIGgraph(eig::EIG; arrow_size=0.03, arrow_os=0.2, width)
#     my_plot = plot(;aspect_ratio=:equal)  # init plot with nothing
        
#     # plot edges
#     for edge in eig.network.edges
        
#         if edge.b == eig.network.v_∞
#             continue  # don't plot v_∞ node
#         end
        
#         # convert int to pair representation for nodes
#         xa, ya = node_int_to_pair(edge.a, width)
#         xb, yb = node_int_to_pair(edge.b, width)
        
#         xs = [xa, xb]
#         ys = [ya, yb]
        
#         plot!(xs, ys; line=(1, :black, :solid), marker=(:circle, :black, 1), label="")
        
#         # draw directed arrowheads
#         if ya == yb
#             # horizontal edge
#             if xa < xb
#                 # arrow pointing right
#                 plot!([xa + arrow_os - arrow_size, xa + arrow_os], [ya + arrow_size, ya]; line=(1, :black, :solid), label="")
#                 plot!([xa + arrow_os - arrow_size, xa + arrow_os], [ya - arrow_size, ya]; line=(1, :black, :solid), label="")
                
#             else
#                 # arrow pointing left
#                 plot!([xa - arrow_os + arrow_size, xa - arrow_os], [ya + arrow_size, ya]; line=(1, :black, :solid), label="")
#                 plot!([xa - arrow_os + arrow_size, xa - arrow_os], [ya - arrow_size, ya]; line=(1, :black, :solid), label="")
#             end
#         elseif xa == xb
#             # vertical edge
#             if ya < yb
#                 # arrow pointing up
#                 plot!([xa - arrow_size, xa], [ya + arrow_os - arrow_size, ya + arrow_os]; line=(1, :black, :solid), label="")
#                 plot!([xa + arrow_size, xa], [ya + arrow_os - arrow_size, ya + arrow_os]; line=(1, :black, :solid), label="")
#             else
#                 # arrow pointing down
#                 plot!([xa - arrow_size, xa], [ya - arrow_os + arrow_size, ya - arrow_os]; line=(1, :black, :solid), label="")
#                 plot!([xa + arrow_size, xa], [ya - arrow_os + arrow_size, ya - arrow_os]; line=(1, :black, :solid), label="")
#             end
#         else
#             # diagonal edge
#             if xa < xb
#                 if ya < yb
#                     # point up-right
#                     plot!([xa + arrow_os - arrow_size, xa + arrow_os], [ya + arrow_os, ya + arrow_os]; line=(1, :black, :solid), label="")
#                     plot!([xa + arrow_os, xa + arrow_os], [ya + arrow_os - arrow_size, ya + arrow_os]; line=(1, :black, :solid), label="")
#                 else
#                     # point down-right
#                     plot!([xa + arrow_os - arrow_size, xa + arrow_os], [ya - arrow_os, ya - arrow_os]; line=(1, :black, :solid), label="")
#                     plot!([xa + arrow_os, xa + arrow_os], [ya - arrow_os + arrow_size, ya - arrow_os]; line=(1, :black, :solid), label="")
                    
#                 end
#             else
#                 if ya < yb
#                     # point up-left
#                     plot!([xa - arrow_os + arrow_size, xa - arrow_os], [ya + arrow_os, ya + arrow_os]; line=(1, :black, :solid), label="")
#                     plot!([xa - arrow_os, xa - arrow_os], [ya + arrow_os - arrow_size, ya + arrow_os]; line=(1, :black, :solid), label="")
#                 else
#                     # point down-left
#                     plot!([xa - arrow_os + arrow_size, xa - arrow_os], [ya - arrow_os, ya - arrow_os]; line=(1, :black, :solid), label="")
#                     plot!([xa - arrow_os, xa - arrow_os], [ya - arrow_os + arrow_size, ya - arrow_os]; line=(1, :black, :solid), label="")
#                 end
#             end
#         end
#     end
    
#     # plot exit nodes (nodes having an arc towards v_∞)
#     for exit_node in eig.network.exit_nodes
#         pair_repr = node_int_to_pair(exit_node, width)
#         plot!(pair_repr; marker=(:circle, :blue, 6), label="")
#     end
    
#     # plot attacker start node
#     plot!(node_int_to_pair(eig.network.v_0, width); marker=(:circle, :red, 4), label="")
    
#     # plot defender start nodes
#     for v_0r in eig.network.v_0r
#         pair_repr = node_int_to_pair(v_0r, width)
#         plot!(pair_repr; marker=(:circle, :lightgreen, 4), label="")
#     end
    
#     display(my_plot)
# end

# # +
# function animateEIGPure(eig::EIG, att_strat, def_strat; arrow_size=0.03, arrow_os=0.2, anim_time=3, fps=30, savefilename="EIG_GIF_EXAMPLE", width)
    
#     num_frames = ceil(Int, anim_time * fps)
    
#     curr_att_num = 1  # attacker state numbers
#     curr_def_nums = fill(1, size(def_strat)[1])  # resource state numbers
    
#     MAX_TIME = eig.t_max  # maximum 'interesting' time in EIG to animate to; choose such that the attacker has entered an exit-node + 1
#     for state in att_strat
#         if state.v in eig.network.exit_nodes
#             MAX_TIME = state.t + 1 # attacker will trivially escape 
#             break;
#         end
#     end
    
    
#     # precompute resource paths
#     res_all_paths = [[] for _ in 1:length(def_strat)]  # ith entry is a list of paths, one for each transition between resource states
#     for (j, res) in enumerate(def_strat)
#         # compute paths for this resource res
#         res_paths = []

#         for (state_num, state) in enumerate(res[1:end-1])
#             # find path from state_num to state_num+1
#             path = shortest_path(eig, state.v, res[state_num+1].v, scale_divisor=eig.speed_D)
#             push!(res_paths, path)
#         end
#         res_all_paths[j] = res_paths
#     end
    
#     anim = @animate for i = 1:num_frames
#         # first determine time t elapsed for EIG as a function of i
#         eig_time = i / num_frames * MAX_TIME
        
#         title_str = string(round(eig_time))
        
#         plot(;aspect_ratio=:equal, title = "t = $title_str", showaxis = false)  # init plot with nothing; also display EIG-time as title
        
#         # plot edges
#         for edge in eig.network.edges

#             if edge.b == eig.network.v_∞
#                 continue  # don't plot v_∞ node
#             end

#             # convert int to pair representation for nodes
#             xa, ya = node_int_to_pair(edge.a, width)
#             xb, yb = node_int_to_pair(edge.b, width)

#             xs = [xa, xb]
#             ys = [ya, yb]

#             plot!(xs, ys; line=(1, :black, :solid), marker=(:circle, :black, 1), label="")

#             # draw directed arrowheads
#             if ya == yb
#                 # horizontal edge
#                 if xa < xb
#                     # arrow pointing right
#                     plot!([xa + arrow_os - arrow_size, xa + arrow_os], [ya + arrow_size, ya]; line=(1, :black, :solid), label="")
#                     plot!([xa + arrow_os - arrow_size, xa + arrow_os], [ya - arrow_size, ya]; line=(1, :black, :solid), label="")

#                 else
#                     # arrow pointing left
#                     plot!([xa - arrow_os + arrow_size, xa - arrow_os], [ya + arrow_size, ya]; line=(1, :black, :solid), label="")
#                     plot!([xa - arrow_os + arrow_size, xa - arrow_os], [ya - arrow_size, ya]; line=(1, :black, :solid), label="")
#                 end
#             elseif xa == xb
#                 # vertical edge
#                 if ya < yb
#                     # arrow pointing up
#                     plot!([xa - arrow_size, xa], [ya + arrow_os - arrow_size, ya + arrow_os]; line=(1, :black, :solid), label="")
#                     plot!([xa + arrow_size, xa], [ya + arrow_os - arrow_size, ya + arrow_os]; line=(1, :black, :solid), label="")
#                 else
#                     # arrow pointing down
#                     plot!([xa - arrow_size, xa], [ya - arrow_os + arrow_size, ya - arrow_os]; line=(1, :black, :solid), label="")
#                     plot!([xa + arrow_size, xa], [ya - arrow_os + arrow_size, ya - arrow_os]; line=(1, :black, :solid), label="")
#                 end
#             else
#                 # diagonal edge
#                 if xa < xb
#                     if ya < yb
#                         # point up-right
#                         plot!([xa + arrow_os - arrow_size, xa + arrow_os], [ya + arrow_os, ya + arrow_os]; line=(1, :black, :solid), label="")
#                         plot!([xa + arrow_os, xa + arrow_os], [ya + arrow_os - arrow_size, ya + arrow_os]; line=(1, :black, :solid), label="")
#                     else
#                         # point down-right
#                         plot!([xa + arrow_os - arrow_size, xa + arrow_os], [ya - arrow_os, ya - arrow_os]; line=(1, :black, :solid), label="")
#                         plot!([xa + arrow_os, xa + arrow_os], [ya - arrow_os + arrow_size, ya - arrow_os]; line=(1, :black, :solid), label="")

#                     end
#                 else
#                     if ya < yb
#                         # point up-left
#                         plot!([xa - arrow_os + arrow_size, xa - arrow_os], [ya + arrow_os, ya + arrow_os]; line=(1, :black, :solid), label="")
#                         plot!([xa - arrow_os, xa - arrow_os], [ya + arrow_os - arrow_size, ya + arrow_os]; line=(1, :black, :solid), label="")
#                     else
#                         # point down-left
#                         plot!([xa - arrow_os + arrow_size, xa - arrow_os], [ya - arrow_os, ya - arrow_os]; line=(1, :black, :solid), label="")
#                         plot!([xa - arrow_os, xa - arrow_os], [ya - arrow_os + arrow_size, ya - arrow_os]; line=(1, :black, :solid), label="")
#                     end
#                 end
#             end
#         end

#         # plot exit nodes (nodes having an arc towards v_∞)
#         for exit_node in eig.network.exit_nodes
#             pair_repr = node_int_to_pair(exit_node, width)
#             plot!(pair_repr; marker=(:circle, :blue, 6), label="")
#         end
        
#         # plot attacker and defenders
        
#         # println("eig_time= $eig_time") 
        
#         # check if attacker has arrived at next node yet
#         while curr_att_num < size(att_strat)[1] && eig_time ≥ att_strat[curr_att_num + 1].t
#             # attacker state has changed; update it
#             curr_att_num += 1
#         end
        
#         # if !(att_strat[curr_att_num].v in eig.network.exit_nodes)  # attacker hasn't reached end node
        
#         if curr_att_num < size(att_strat)[1] - 1  # attacker hasn't reached an exit node (a 2nd last state must be an exit node)
#             # plot attacker point; find vector from current node to next node, and scale by time
#             t = (eig_time - att_strat[curr_att_num].t) / (att_strat[curr_att_num + 1].t - att_strat[curr_att_num].t)
#             node_a = node_int_to_pair(att_strat[curr_att_num].v, width)
#             node_b = node_int_to_pair(att_strat[curr_att_num + 1].v, width)

#             point_to_plot = ((1 - t) .* node_a) .+ (t .* node_b)

#             # display(point_to_plot)

#             plot!(point_to_plot; marker=(:circle, :red, 4), label="")
#         else
#             # println("curr_att_num= $curr_att_num")
#         end

#         # plot defender nodes
        
#         for (j, res) in enumerate(def_strat)
#             # check if defender resource has left the current node
#             while curr_def_nums[j] < size(res)[1] && eig_time > res[curr_def_nums[j]].t_b
#                 # update resource state
#                 curr_def_nums[j] += 1
#             end
            
#             # check if resource is in transit, or at a node
            
#             if res[curr_def_nums[j]].t_a ≤ eig_time ≤ res[curr_def_nums[j]].t_b
#                 # plot stationary resource at node
#                 pair_repr = node_int_to_pair(res[curr_def_nums[j]].v, width)
#                 plot!(pair_repr; marker=(:circle, :lightgreen, 4), label="")
#             else
#                 # TODO, in transit to next node (which may not be a neighbourhood node)
                
#                 # find shortest path from current resource node to next node (currently inefficient - multiple calls for the same )
                
#                 # display(res_all_paths)
#                 # display(curr_def_nums)
#                 # display(j)
                
#                 res_paths = res_all_paths[j]  # all paths of resource (kth entry is path from state k to k+1)
                
#                 path = res_paths[curr_def_nums[j] - 1]  # current path in transit
                
#                 # find neighbouring nodes in path such that time in path
#                 # [(v_0, 0), (v_1, 5), (v_2, 7), (v_3, 10)]
                
#                 # display(path)
#                 # display(res[curr_def_nums[j] - 1])
                
#                 edge_start = 0
#                 edge_end = 0  # node of the end of the edge currently being traversed
#                 time_start = 0
#                 time_end = eig.t_max
#                 for idx in 1:length(path)
#                     if path[idx].t > eig_time - res[curr_def_nums[j] - 1].t_b
#                         # found edge to traverse
                        
#                         # println("Found idx=", idx, " path[idx].t = ", path[idx].t, " eig_time - res[curr_def_nums[j] - 1].t_b = ", eig_time - res[curr_def_nums[j] - 1].t_b)
#                         # println("eig_time = ", eig_time)
                        
#                         edge_start = path[idx - 1].v
#                         edge_end = path[idx].v
                        
#                         time_start = res[curr_def_nums[j] - 1].t_b + path[idx - 1].t  # eig time corresponding to resource being precisely at edge_start
#                         time_end = res[curr_def_nums[j] - 1].t_b + path[idx].t
#                         # println("time_start=$time_start, time_end=$time_end")
#                         break;
#                     end
#                 end
                
#                 # plot defender point; find vector from current node to next node, and scale by t (LERP)
#                 t = (time_end - eig_time) / (time_end - time_start)
                
#                 if t > 1
#                     println("!!!!!!!!!!!!! t = $t")
#                 end
                
#                 # display(t)
                
#                 node_b = node_int_to_pair(edge_start, width)  # !! swapped b with a, but it seems to fix the 'reverse' bug
#                 node_a = node_int_to_pair(edge_end, width)

#                 point_to_plot = ((1 - t) .* node_a) .+ (t .* node_b)

#                 # display(point_to_plot)

#                 plot!(point_to_plot; marker=(:circle, :green, 4), label="")
#             end
#         end
        
        
# #         for v_0r in eig.network.v_0r
# #             pair_repr = node_int_to_pair(v_0r, width)
# #             plot!(pair_repr; marker=(:circle, :lightgreen, 4), label="")
# #         end
#     end
    
#     gif(anim, savefilename * ".gif", fps=fps)
    
#     # display(my_plot)
# end
# # -

# #### Interdiction Check & CoreLP

# +
function doesinterdict(att, def; printing=false)
    """
    Given pure attacker and defender strategies, returns True if an interdiction occurs, False otherwise.
    """
    # display("DOES_INT")
    # display(att)
    # display(def)
    
    if printing
        println("Doing interdiction check of att_strat")
        display(att)
        println("def_strat:")
        display(def)
    end
    
    for res_schedule in def
        for r_st in res_schedule
            for a_st in att
                if a_st.v == r_st.v && r_st.t_a ≤ a_st.t ≤ r_st.t_b
                    if printing
                        println("Found interdiction at node $(a_st.v) with r_st.t_a=$(r_st.t_a) and a_st.t=$(a_st.t) and r_st.t_b=$(r_st.t_b)")
                    end
                    return true  # interdiction
                end
            end
        end
    end
    
    return false
    
    
    
    for res_schedule in def 
        # check if interdiction occurs for this resource (i.e. ∃j,i s.t. t_j^r- ≤ t_i ≤ t_j^r+ and v_i = v_j^r)
        res_state_num = 1
        att_state_num = 1
        
        while true
            a_st = att[att_state_num]
            r_st = res_schedule[res_state_num]
            
            # check if current attacker arrival time is within the 'block-time' of the current resource state
            if r_st.t_a ≤ a_st.t ≤ r_st.t_b
                # check if at same node
                if a_st.v == r_st.v
                    return true  # interdiction
                else
                    # update to next attacker state
                    if att_state_num ≥ length(att); break; end  # no more states to check
                    att_state_num += 1
                end
            else
                # attacker arrived after resource left; update to next resource state
                if res_state_num ≥ length(res_schedule); break; end  # no more states to check
                res_state_num += 1
            end
        end
    end
    return false  # no interdiction
end


# # testing
# att = [(v=1,t=0), (v=3, t=3.8), (v=-1, t=5)]
# def = [
#     [(v=2, t_a=0, t_b=2), (v=3, t_a=4, t_b=5)],  # resource 1 schedule
#     [(v=2, t_a=0, t_b=4)]  # resource 2 schedule
# ]

# doesinterdict(att, def)

# +
# # compute interdiction matrix using doesinterdict (attacker chooses row)
# # intdict = [doesinterdict(att_strats[a], def_strats[d]) for a=1:length(att_strats), d=1:length(def_strats)]
# int_dict = [
#     0 1 1 1;
#     1 1 0 0;
#     1 0 1 0;
#     1 0 0 1;
# ]

# # int_dict = rand([0, 1], 5, 5)

# display("intdict matrix computed")
# display(int_dict)

# LP = Model(Gurobi.Optimizer);
# # set_silent(Gurobi);

# @variable(LP, v)  # objective function variable
# @variable(LP, y[d=1:size(int_dict)[2]] ≥ 0)  # variable for each defender strategy

# @constraint(LP, strat_con[a=1:size(int_dict)[1]], sum(int_dict[a, d] * y[d] for d=1:size(int_dict)[2]) ≥ v)
# @constraint(LP, prob_con, sum(y) == 1)

# @objective(LP, Max, v)

# optimize!(LP)

# # get dual solution (for attacker strategy)
# att_probs = [dual(strat_con[a]) for a=1:size(int_dict)[1]]
# def_probs = value.(LP[:y])
# obj_value = objective_value(LP)

# +
function coreLP(att_strats, def_strats, intdict_mtx; optimiser=Gurobi.Optimizer, silent=true)
    """
    Given available attacker strategies and defender strategies, finds an optimal mixed strategy solution for both attacker and defender.
    Uses the full LP formulation for finding a (mixed) Nash equilibrium.
    
    att_strats: list of available attacker strategies
    def_strats: list of available defender strategies
    intdict_mtx: Binary matrix with entry [i,j] indicating whether att strat i and def strat j give interdiction 
    """
#     # compute interdiction matrix using doesinterdict (attacker chooses row); can be more efficient by only computing the new possibilities in each EIGzhang iteration
#     intdict_mtx = [doesinterdict(att_strats[a], def_strats[d]) for a=1:length(att_strats), d=1:length(def_strats)]
    
#     display("intdict matrix computed")
#     display(intdict_mtx)
    
    LP = Model(optimiser);
    if silent; set_silent(LP); end
    # set_silent(Gurobi);
    
    @variable(LP, v)  # objective function variable
    @variable(LP, y[d=1:length(def_strats)] ≥ 0)  # variable for each defender strategy
    
    @constraint(LP, strat_con[a=1:length(att_strats)], sum(intdict_mtx[a,d] * y[d] for d=1:length(def_strats)) ≥ v)
    @constraint(LP, prob_con, sum(y) == 1)
    
    @objective(LP, Max, v)
    
    
#     @variable(LP, w)  # objective function variable
#     @variable(LP, x[a=1:length(att_strats)] ≥ 0)  # variable for each attacker strategy
    
#     @constraint(LP, strat_con[d=1:length(def_strats)], sum(intdict[a,d] * x[a] for a=1:length(att_strats)) ≤ w)
#     @constraint(LP, prob_con, sum(x) == 1)
    
#     @objective(LP, Min, w)

    optimize!(LP)
    
    if termination_status(LP) != OPTIMAL
        error("CoreLP did not solve to optimality. ", termination_status(LP))
    end
    
    # get dual solution (for attacker strategy)
    att_probs = [dual(strat_con[a]) for a=1:length(att_strats)]
    def_probs = value.(LP[:y])
    obj_value = objective_value(LP)
    
    return att_probs, def_probs, obj_value
end

# +
# att_strat_1 = [(v=1, t=0), (v=2, t=1.5), (v=-1, t=5)]
# att_strat_2 = [(v=1, t=0), (v=3, t=2), (v=-1, t=5)]
# att_strat_3 = [(v=1, t=0), (v=2, t=1), (v=-1, t=5)]
# def_strat_1 = [
#     [(v=2, t_a=0, t_b=2), (v=3, t_a=4, t_b=5)],  # resource 1 schedule
#     [(v=4, t_a=0, t_b=1)]  # resource 2 schedule
# ]
# def_strat_2 = [
#     [(v=2, t_a=0, t_b=1), (v=3, t_a=4, t_b=5)],  # resource 1 schedule
#     [(v=4, t_a=0, t_b=1), (v=3, t_a=2, t_b=5)]  # resource 2 schedule
# ]
# def_strat_3 = [
#     [(v=2, t_a=0, t_b=1), (v=3, t_a=4, t_b=5)],  # resource 1 schedule
#     [(v=4, t_a=0, t_b=1), (v=6, t_a=2, t_b=5)]  # resource 2 schedule
# ]


# # convert to list of strategies
# att = [att_strat_1, att_strat_2, att_strat_3]
# def = [def_strat_1, def_strat_2, def_strat_3]

# intdict_mtx = intdict_mtx = [doesinterdict(att[a], def[d]) for a=1:length(att), d=1:length(def)]

# att_probs, def_probs, obj_value = coreLP(att, def, intdict_mtx)
# -

# #### Zhang et al. Double Oracle

function EIGSzhang!(eig::EIG, att_strats, def_strats; L_max=5, DO=defenderoraclezhang, AO=attackeroraclezhang, max_iters=Inf, printing=1,
        total_timeout=300, A_timeout=60.0, D_timeout=60.0, silent_solvers=false, abstol=1e-6)
    """ TODO: bounds given by oracles - actually, just giving the data of the oracle objectives every run is enough (we can compute bounds easily from this data)
    Implements algorithm 1 in the Zhang paper (the main double oracle).
    Given initial attacker and defender strategies, generates new pure strategies until no improving strategies are found
    
    att_strats: list of initial attacker strategies (will be added to)
    def_strats: list of initial defender strategies (will be added to)
    total_timeout: Time duration (secs) given to the entire run in total (at least the main while loop).
    A_timeout: Time duration (secs) given to each attacker oracle call (timeout the oracle if exceeded)
        If nothing, then computes timeout for AO based on total_timeout and remaining time
    D_timeout: Time duration (secs) given to each defender oracle call (timeout the oracle if exceeded)
        If nothing, then computes timeout for DO based on total_timeout and remaining time.
    printing: An Int within {0, 1, 2}:
        0 is no printing whatsoever
        1 is some printing (only initiating LPs, and whether improving strategies are found or not each iteration)
        2 is full printing
    silent_solvers: Bool for if the LP solvers should be silent or not
    abstol: Absolute tolerance for an objective to be considered a strict improvement
    """
    convergence = false;
    
    obj_core_over_time = []
    obj_DO_over_time = []
    obj_AO_over_time = []
    
    DO_time_over_time = []  # time taken to run the DO in each iteration
    AO_time_over_time = []
    coreLP_time_over_time = []

    AO_network_nodes_over_time = []  # number of nodes in the attacker oracle graph constructed (support-based construction)
    AO_network_edges_over_time = []  # number of arcs in the attacker oracle graph constructed (support-based construction)
    DO_network_nodes_over_time = []  # number of nodes in the defender oracle graph constructed (support-based construction)
    DO_network_edges_over_time = []  # number of arcs in the defender oracle graph constructed (support-based construction)
    
    iter_counter = 1

    # construct graphs if using network oracles
    if AO == attackeroraclenew
        # if printing ≥ 1
        #     println("Constructing attacker oracle graph")
        #     flush(stdout)
        # end
        # time_graph_att_constr = @elapsed new_graph_att = constructattackeroraclegraph(eig)
        # if printing ≥ 1
        #     println("Finished attacker oracle graph")
        #     flush(stdout)
        # end
        time_graph_att_constr = []
    else
        time_graph_att_constr = nothing  # for consistent returning purposes
        # new_graph_att = nothing
    end

    if DO == defenderoraclenew
        # if printing ≥ 1
        #     println("Constructing defender oracle graph")
        #     flush(stdout)
        # end
        # time_graph_def_constr = @elapsed new_graph_def = constructdefenderoraclegraph(eig)
        # if printing ≥ 1
        #     println("Finished defender oracle graph")
        #     flush(stdout)
        # end
        time_graph_def_constr = []
    else
        time_graph_def_constr = nothing  # for consistent returning purposes
        # new_graph_def = nothing
    end
    
    # will be updated each iteration (just before coreLP run)
    intdict_mtx = [doesinterdict(att_strats[a], def_strats[d]) for a=1:length(att_strats), d=1:length(def_strats)]
    
    # timeout
    total_timeout_ns = total_timeout * 1e9  # timeout in nanoseconds
    start_time_ns = time_ns()  # current time in nanoseconds (1e-9 secs)
    while !convergence && iter_counter < max_iters && time_ns() - start_time_ns < total_timeout_ns
        convergence = true;  # assume true until an improving strategy is found
        
        if printing ≥ 1; println("\n!!!!!!!!!!!!STARTING CORELP"); end
        # if printing
        #     println("\n!!!!!!!!!!!!STARTING CORELP")
        # end
        # find optimal solution with current strats using coreLP; outputs probability of interdiction
        results_ = @timed coreLP(att_strats, def_strats, intdict_mtx, silent=silent_solvers)
        att_probs, def_probs, obj_core = results_.value
        push!(coreLP_time_over_time, results_.time)
        
        if printing ≥ 2
            println("coreLP results:\natt_probs:")
            display(att_probs)
            println("\natt_strats:")
            display(att_strats)
            println("\ndef_probs:")
            display(def_probs)
            println("\ndef_strats")
            display(def_strats)
        end
        
        if obj_core == 1
            # probability of interdiction is 1, so no improving defender strategy exists
            if printing ≥ 1; println("\n!!!!!!!!!!!! SKIPPING DEFENDER ORACLE, obj_core = $obj_core"); end
            # if printing
            #     println("\n!!!!!!!!!!!! SKIPPING DEFENDER ORACLE, obj_core = $obj_core")
            # end
            obj_DO = nothing
            push!(DO_time_over_time, nothing)
        else
            if printing ≥ 1; println("\n!!!!!!!!!!!! STARTING DEFENDER ORACLE"); end
            # if printing
            #     println("\n!!!!!!!!!!!! STARTING DEFENDER ORACLE")
            # end
            
            # compute timeout time (secs)
            # println("outside, the D_timeout is $D_timeout")
            if isnothing(D_timeout)
                D_timeout_ = (start_time_ns + total_timeout_ns - time_ns()) / 1e9  # time given (secs) to defender oracle
                println("the D_timeout_ is $D_timeout_ (set)")
                flush(stdout)

                if D_timeout_ <= 0
                    # timeout, break
                    println("the D_timeout_ is negative, breaking")
                    flush(stdout)
                    convergence = false
                    break;
                end
            else
                D_timeout_ = D_timeout
            end
            
            results_ = @timed DO(eig, att_strats, att_probs, timeout=D_timeout_, silent=silent_solvers)
            new_def_strat = results_.value.new_def_strat
            obj_DO = results_.value.obj_value
            LP_DO = results_.value.LP_DO
            
            push!(DO_time_over_time, results_.time)
            
            # get size of defender network graph (if done)
            if DO == defenderoraclenew
                push!(DO_network_nodes_over_time, results_.value.num_nodes_new_graph)
                push!(DO_network_edges_over_time, results_.value.num_edges_new_graph)
                push!(time_graph_def_constr, results_.value.time_new_graph_construction)
            end

            # # return signature of defenderoraclenew 
            # return (new_def_strat=new_def_strat,
            #     obj_value=obj_value,
            #     LP_DO=LP_DOn,
            #     num_nodes_new_graph=num_nodes_new_graph,
            #     num_edges_new_graph=num_edges_new_graph,
            #     time_new_graph_construction=time_new_graph_construction)
            
            # new_def_strat, obj_DO, LP_DO = DO(eig, att_strats, att_probs, L_max=L_max)
            
            if primal_status(LP_DO) != FEASIBLE_POINT
                # oracle didn't find a feasible point
                convergence = false
                if printing ≥ 1; println("\n!!! NO FEASIBLE DEFENDER STRATEGY FOUND (DO TIMEOUT of $D_timeout_ )"); end
            elseif !isnothing(obj_DO)
                if obj_DO > obj_core + abstol
                    # defender oracle found an improving strategy
                    push!(def_strats, new_def_strat)
                    push!(def_probs, 0)  # play this strategy with 0 probability - for the attacker oracle in this iteration
                    
                    # update intdict_mtx (new column for defender)
                    new_col = [doesinterdict(a, new_def_strat) for a in att_strats]
                    intdict_mtx = [intdict_mtx;; new_col]  # concat new col
                    
                    convergence = false
                    if printing ≥ 2
                        println("\n!!! NEW DEFENDER STRAT:")
                        display(def_strats[end])
                        println("\n def_probs (before running coreLP)")
                        display(def_probs)
                    end
                else
                    # objective found, but not within tolerance; leave convergence as true
                    if printing ≥ 1; println("\n!!! NO SUFFICIENTLY (abstol=$abstol) IMPROVING DEFENDER STRATEGY FOUND"); end
                end
            elseif termination_status(LP_DO) != OPTIMAL
                convergence = false
                if printing ≥ 1; println("\n!!! NO IMPROVING DEFENDER STRATEGY FOUND (DO TIMEOUT of $D_timeout_ )"); end
            else
                if printing ≥ 1; println("\n!!! NO IMPROVING DEFENDER STRATEGY PROVED; obj_DO = $obj_DO"); end  # optimal point found and couldn't improve
                
                # if printing
                #     println("\n!!! NO IMPROVING DEFENDER STRATEGY FOUND; obj_DO = $obj_DO")
                # end
            end
        end
        
        if obj_core == 0
            # probability of interdiction is 0, so no improving attacker strategy exists
            if printing ≥ 1; println("\n!!!!!!!!!!!! SKIPPING ATTACKER ORACLE, obj_core = $obj_core"); end
            # if printing
            #     println("\n!!!!!!!!!!!! SKIPPING ATTACKER ORACLE, obj_core = $obj_core")
            # end
            obj_AO = nothing
            push!(AO_time_over_time, nothing)
        else
            if printing ≥ 1; println("\n!!!!!!!!!!!! STARTING ATTACKER ORACLE"); end
            # if printing
            #     println("\n!!!!!!!!!!!! STARTING ATTACKER ORACLE")
            # # display(def_strats)
            # # display(def_probs)
            # end
            
            # println("outside, the A_timeout is $A_timeout")
            if isnothing(A_timeout)
                A_timeout_ = (start_time_ns + total_timeout_ns - time_ns()) / 1e9  # time given (secs) to attacker oracle
                println("the A_timeout_ is $A_timeout_ (set)")
                flush(stdout)
                
                if A_timeout_ <= 0
                    # timeout, break
                    println("the A_timeout_ is negative, breaking")
                    flush(stdout)
                    convergence = false
                    break;
                end
            else
                A_timeout_ = A_timeout
            end
            
            results_ = @timed AO(eig, def_strats, def_probs, timeout=A_timeout_, silent=silent_solvers)
            # new_att_strat, obj_AO, LP_AO = results_.value
            
            new_att_strat = results_.value.new_att_strat
            obj_AO = results_.value.obj_value
            LP_AO = results_.value.LP_AO
            
            push!(AO_time_over_time, results_.time)
            
            # get size of attacker network graph (if done)
            if AO == attackeroraclenew
                push!(AO_network_nodes_over_time, results_.value.num_nodes_new_graph)
                push!(AO_network_edges_over_time, results_.value.num_edges_new_graph)
                push!(time_graph_att_constr, results_.value.time_new_graph_construction)
            end
            
            # new_att_strat, obj_AO, LP_AO = AO(eig, def_strats, def_probs, L_max=L_max)
            if primal_status(LP_AO) != FEASIBLE_POINT
                # oracle didn't find a feasible point
                convergence = false
                if printing ≥ 1; println("\n!!! NO FEASIBLE ATTACKER STRATEGY FOUND (AO TIMEOUT of $A_timeout_ )"); end
            elseif !isnothing(obj_AO)
                if obj_AO < obj_core - abstol
                    # attacker oracle found an improving strategy
                    push!(att_strats, new_att_strat)
                    convergence = false
                    
                    # update intdict_mtx (new row for attacker)
                    new_row = [doesinterdict(new_att_strat, d) for d in def_strats]
                    intdict_mtx = [intdict_mtx; transpose(new_row)]  # concat new row
                    
                    if printing ≥ 2
                        println("\n!!! NEW ATTACKER STRAT:")
                        display(att_strats[end])
                        println("\n att_probs (before running coreLP):")
                        display(att_probs)
                    end
                else
                    # found objective value, but not within tolerance; leave convergence as true
                    if printing ≥ 1; println("\n!!! NO SUFFICIENTLY (abstol=$abstol) IMPROVING ATTACKER STRATEGY FOUND"); end
                end
            elseif termination_status(LP_AO) != OPTIMAL
                convergence = false
                if printing ≥ 1; println("\n!!! NO IMPROVING ATTACKER STRATEGY FOUND (AO TIMEOUT of $A_timeout_ )"); end
            else
                if printing ≥ 1; println("\n!!! NO IMPROVING ATTACKER STRATEGY FOUND; obj_AO = $obj_AO"); end
            end
        end
        
        push!(obj_core_over_time, obj_core)
        push!(obj_DO_over_time, obj_DO)
        push!(obj_AO_over_time, obj_AO)
        
        iter_counter += 1
    end
    
    convergence_flag = true
    
    if iter_counter == max_iters
        if printing ≥ 1; println("!!! Reached max iterations $max_iters"); end
        convergence_flag = false
    elseif time_ns() - start_time_ns >= total_timeout_ns
        if printing ≥ 1; println("!!! Reached total timeout for EIGzhang double oracle"); end
        convergence_flag = false
    else
        if printing ≥ 1; println("Found sub-game NE"); end
    end
    
    # do one last coreLP to get probs in right format
    att_probs, def_probs, obj_core = coreLP(att_strats, def_strats, intdict_mtx, silent=silent_solvers)
    
    return (
        att_strats=att_strats,
        att_probs=att_probs,
        def_strats=def_strats,
        def_probs=def_probs,
        obj_core_over_time=obj_core_over_time,
        obj_DO_over_time=obj_DO_over_time,
        obj_AO_over_time=obj_AO_over_time,
        coreLP_time_over_time=coreLP_time_over_time,
        DO_time_over_time=DO_time_over_time,
        AO_time_over_time=AO_time_over_time,
        convergence_flag=convergence_flag,
        time_graph_att_constr=time_graph_att_constr,
        time_graph_def_constr=time_graph_def_constr,
        AO_network_nodes_over_time=AO_network_nodes_over_time,
        AO_network_edges_over_time=AO_network_edges_over_time,
        DO_network_nodes_over_time=DO_network_nodes_over_time,
        DO_network_edges_over_time=DO_network_edges_over_time
    )
end

# #### Original Zhang et al. Oracles

# +
function defenderoraclezhang(eig::EIG, att_strats, att_probs; L_max=nothing, optimiser=Gurobi.Optimizer, timeout=60.0, silent=true, printing=0)
    """
    Given available attacker strategies, finds an optimal pure defender strategy.
    
    att_strats: list of available attacker strategies
    att_probs: list of probabilities of each attacker strategy (given by coreLP) 
    graph: Does nothing for this function (is a parameter here so that defender oracles have a consistent format)
    
    nodes, edges, v_0, v_0r, exit_nodes, T_MAX, DELTA
    
    Return format:
        new_def_strat, obj_value, LP_DO
    """
    # L_max very haphazard choice - ??!! Should we also be trying to minimise useful states?
    
    T_MAX = eig.t_max
    DELTA = eig.δ
    
    # estimate upper bound L_max on number of resource states for any resource 
    if isnothing(L_max)
        L_max = eig.network.num_nodes  # haphazard choice for L_max (we have no theory on whether the defender should backtrack or not)
    end
    
    bigM = ceil(T_MAX) + 1  # should be enough for the big M constraints (14 & 15)

    nodes = 1:eig.network.num_nodes
    nodes_1 = 1:eig.network.num_nodes-1  # all nodes except v_∞
    
    v_0r = eig.network.v_0r  # defender start nodes
    v_0 = eig.network.v_0
    
    is_path_mtx = [eig.dist_mtx[i, j] < Inf ? 1 : 0 for i=nodes, j=nodes]  # ijth entry is 1 if there is a path from i to j, 0 otherwise
    
    # display(length(v_0r))
    # display(length(att_strats))
    # display(att_strats[1])
    
    # r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])
    LP_DO = Model(optimiser);
    if silent; set_silent(LP_DO); end
    
    @variable(LP_DO, s[r=1:length(v_0r), i=1:L_max, v=nodes], Bin)  # Ind. for d_r at node v at time i (in d_r's schedule)
    @variable(LP_DO, w[r=1:length(v_0r), i=1:L_max, v=nodes_1, u=nodes_1], Bin)  # Ind. if both s_{i,v}^r and s_{i+1,u}^r are true (so if u-v is the path taken by d_r between (their) state i and i+1)
    @variable(LP_DO, a[r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])], Bin)  # Ind. for if attacker arrives at their node j after d_r arrives at i (so t_i < t_j aka t_i - t_j < 0)
    @variable(LP_DO, b[r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])], Bin)  # Ind. for if attacker arrives at their node j before d_r arrives at i
    @variable(LP_DO, c[r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])], Bin)  # Ind. for if interdiction occurs at attacker node j (using strat A), defender resource r at node i (in their schedule)
    @variable(LP_DO, z[A=1:length(att_strats)], Bin)
    
    @variable(LP_DO, k[r=1:length(v_0r), i=1:L_max] ≥ 0, Int)  # multiplier of DELTA for resource r at state i (restricting to ≥ 1 could make the LP infeasible)
    @variable(LP_DO, 0 ≤ t_in[r=1:length(v_0r), i=1:L_max] ≤ ceil(T_MAX))  # ceiling for integral EIG (unecessary here?)
    @variable(LP_DO, 0 ≤ t_ou[r=1:length(v_0r), i=1:L_max] ≤ ceil(T_MAX))  # ceiling for integral EIG
    # @variable(LP_DO, 0 ≤ t_ou[r=1:length(v_0r), i=1:L_max] ≤ T_MAX + DELTA)  # !! changed from T_MAX to T_MAX + DELTA for feasibility
    
    @constraint(LP_DO, con9a[r=1:length(v_0r)], s[r, 1, v_0r[r]] == 1)
    @constraint(LP_DO, con9b[r=1:length(v_0r), i=1:L_max], sum( s[r, i, v] for v=nodes_1 ) == 1)
    @constraint(LP_DO, con10a[r=1:length(v_0r), i=1:L_max, v=nodes_1, u=nodes_1], w[r, i, v, u] ≤ s[r, i, v])
    @constraint(LP_DO, con10b[r=1:length(v_0r), i=1:L_max-1, v=nodes_1, u=nodes_1], w[r, i, v, u] ≤ s[r, i+1, u])
    @constraint(LP_DO, con11[r=1:length(v_0r), i=1:L_max-1, v=nodes_1, u=nodes_1], w[r, i, v, u] ≥ s[r, i, v] + s[r, i+1, u] - 1)
    @constraint(LP_DO, con12a[r=1:length(v_0r)], t_in[r, 1] == 0)
    @constraint(LP_DO, con12b[r=1:length(v_0r)], t_ou[r, L_max] == ceil(T_MAX))  # ceiling for integral EIG
    @constraint(LP_DO, con12c[r=1:length(v_0r), i=1:L_max-1], t_ou[r, i] == t_in[r, i] + k[r, i]*DELTA)  # !! don't include last state for feasibility
    
    # @constraint(LP_DO, con12d[r=1:length(v_0r), i=1:L_max], t_ou[r, i] == t_in[r, i] + k[r, i]*DELTA)
    
    @constraint(LP_DO, con13[r=1:length(v_0r), i=1:L_max-1], t_in[r, i+1] == t_ou[r, i] + sum(eig.dist_mtx[v, u] * w[r, i, v, u] / eig.speed_D
            for u=nodes_1 for v=nodes_1 if eig.dist_mtx[v, u] != Inf))  # dist_mtx[v, u] == inf iff there is no vu-path)
    # @constraint(LP_DO, con13[r=1:length(v_0r), i=1:L_max-1], t_in[r, i+1] == t_ou[r, i] + sum(dist_mtx[v, u]*w[r, i, v, u] for u=nodes_1 for v=nodes_1))  # if dist_mtx[v, u] = -inf (meaning there is no uv-path)
    
    @constraint(LP_DO, con13p[r=1:length(v_0r), i=1:L_max-1, v=nodes_1, u=nodes_1], w[r, i, v, u] ≤ is_path_mtx[v, u])  # !! new constraint? if w[r,i,v,u] = 1 then there must be a vu-path
    
    # idea: add dummy states for defender ? - No, dist_mtx[v, v] is 0, so defender states can stay at the same node between states spending 0 time
    # symmetry breaking constraint - if a defender visits the same node between two consecutive states then they must stay at that node forever
    # @constraint(LP_DO, consym[r=1:length(v_0r), i=1:L_max-1, v=nodes_1], s[r, i, v] + s[r, i+1, v] ≥ 2)
    
    @constraint(LP_DO, con14a[r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])], -bigM * a[r, i, A, j] ≤ t_in[r, i] - att_strats[A][j].t)
    @constraint(LP_DO, con14b[r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])], t_in[r, i] - att_strats[A][j].t ≤ bigM * (1 - a[r, i, A, j]))
    
    @constraint(LP_DO, con15a[r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])], -bigM * b[r, i, A, j] ≤ att_strats[A][j][2] - t_ou[r, i])
    @constraint(LP_DO, con15b[r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])], att_strats[A][j].t - t_ou[r, i] ≤ bigM * (1 - b[r, i, A, j]))
    
    @constraint(LP_DO, con16[r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])], c[r, i, A, j] ≤ ( a[r, i, A, j] + b[r, i, A, j] + s[r, i, att_strats[A][j].v] ) / 3)
    @constraint(LP_DO, con17[r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])], c[r, i, A, j] ≥ a[r, i, A, j] + b[r, i, A, j] + s[r, i, att_strats[A][j].v] - 2)
    @constraint(LP_DO, con18[A=1:length(att_strats)], z[A] ≤ sum(c[r, i, A, j] for j=1:length(att_strats[A]) for r=1:length(v_0r) for i=1:L_max))
    
    # !! problem: s variable allows s[r, i, v_∞] = 1, but defender cannot go on v_∞.
    # Fix s[r, i, v_∞] = 1 (since defender is not allowed to go on v_∞)
    for r=1:length(v_0r)
        for i=1:L_max
            fix(s[r, i, eig.network.v_∞], 0; force=true)
        end
    end
    
    # @objective(LP_DO, Max, - sum( (1 - z[A])*att_probs[A] for A=1:length(att_strats) ))  # original objective in Zhang paper
    @objective(LP_DO, Max, sum( z[A]*att_probs[A] for A=1:length(att_strats) ))  # new objective that directly outputs probability of interdiction
    
    set_time_limit_sec(LP_DO, timeout)

    # start_time_yeet = time_ns()
    
    optimize!(LP_DO)

    # final_time_yeet = time_ns()
    # duration_yeet = (final_time_yeet - start_time_yeet) / 1e9

    # println("! Zhang DO took $duration_yeet secs to solve")
    
    if termination_status(LP_DO) == TIME_LIMIT
        if primal_status(LP_DO) != FEASIBLE_POINT
            println("Zhang DO timedout after ", timeout, " secs, but didn't find a feasible point")
            flush(stdout)
            # do something here? Like give it a bit more time to find a feasible point?
            return (
                new_def_strat=nothing,
                obj_value=nothing,
                LP_DO=LP_DO        
            )
            # return nothing, nothing, LP_DO  # return essentially nothing
        elseif primal_status(LP_DO) == FEASIBLE_POINT
            println("Zhang DO timedout after ", timeout, " secs, found a feasible point")
            flush(stdout)
        end
    else
        # no timeout
        if termination_status(LP_DO) != OPTIMAL
            error("Zhang DO did not solve to optimality and did not timeout. ", termination_status(LP_DO))
        end
    end
    
    # extract defender strategy - given by s, t_in, t_ou variables
    new_def_strat = []
    for r=1:length(v_0r)
        # construct resource schedule
        res_schedule = []
        
        curr_v = v_0r[r]  # start node
        curr_t_a_value = 0  # start at 0
        curr_t_b_value = value(t_ou[r, 1])
        for i=1:L_max
            # find vertex v such that s[r, i, v] = 1
            vertex = -1
            # println("computing r=$r and i=$i")
            
            for v=nodes
                # println("value(s[r, i, v]) = ", value(s[r, i, v]))
                if value(s[r, i, v]) > 0.5
                    vertex = v
                    break;
                end
            end

            if length(res_schedule) == 0
                if vertex == curr_v
                    # still looking for that first state; update t_b value
                    curr_t_b_value = value(t_ou[r, i])
                else
                   # first state found
                    push!(res_schedule, (v=curr_v, t_a=curr_t_a_value, t_b=curr_t_b_value))

                    curr_t_a_value = value(t_in[r, i])
                    curr_t_b_value = value(t_ou[r, i])
                    curr_v = vertex
                end
            else
                # there are some states already
                if vertex == curr_v
                    # same vertex as previous; update t_b value
                    curr_t_b_value = value(t_ou[r, i])
                else
                    # node is different to previous; update all values
                    push!(res_schedule, (v=curr_v, t_a=curr_t_a_value, t_b=curr_t_b_value))
    
                    curr_t_a_value = value(t_in[r, i])
                    curr_t_b_value = value(t_ou[r, i])
                    curr_v = vertex
                end
            end

            
            # if length(res_schedule) == 0 || vertex == res_schedule[end].v
            #     # still looking for that first state or same vertex as previous; update t_b value
            #     curr_t_b_value = value(t_ou[r, i])
            # else
            #     # node is different to previous; update all values
            #     push!(res_schedule, (v=curr_v, t_a=curr_t_a_value, t_b=curr_t_b_value))

            #     curr_t_a_value = value(t_in[r, i])
            #     curr_t_b_value = value(t_ou[r, i])
            #     curr_v = vertex
            # end
            
            # push!(res_schedule, (v=vertex, t_a=value(t_in[r, i]), t_b=value(t_ou[r, i])))
        end
        push!(new_def_strat, res_schedule)
    end
    
    obj_value = objective_value(LP_DO)
    
    # TESTING
    
#     # interdict test
#     intdict = [doesinterdict(att_strats[a], new_def_strat, printing=true) for a=1:length(att_strats)]
    
#     for a=1:length(att_strats)
#         println("z value for a=$a is $(value(z[a]))")        
#         if (value(z[a]) >= 0.5) && intdict[a] == 0 && att_probs[a] > 0
#             # z variable says interdiction occurs against attacker strat A, but doesinterdict says no interdiction
#             println("\n?????? For att_strat No. $a, z variable is 1, but doesinterdict says interdiction OCCURS NOT")
#         end
#         if (value(z[a]) <= 0.5) && intdict[a] == 1 && att_probs[a] > 0
#             println("\n?????? For att_strat No. $a, z variable is 0, but doesinterdict says interdiction OCCURS")
            
#             if att_probs[a] == 0
#                 println("Nevermind, attacker plays $a with prob 0")
#             end
#         end
#     end

#     # print out w = 1 variables (r, i, v, u)
#     for r in 1:length(v_0r)
#         for i in 1:L_max-1
#             for v in nodes_1
#                 for u in nodes_1
#                     if value(w[r, i, v, u]) >= 0.5
#                         println("   Found w[r,i,v,u]=1 with r,i,v,u = $r, $i, $v, $u")
#                     end
#                 end
#             end
#         end
#     end
    
#     # print out s = 1 variables (r, i, v)
#     for r in 1:length(v_0r)
#         for i in 1:L_max-1
#             for v in nodes
#                 if value(s[r, i, v]) >= 0.5
#                     println("   Found s[r, i, v]=1 with r,i,v = $r, $i, $v")
#                 end
#             end
#         end
#     end
    
#     for r in 1:length(v_0r)
#         for i=1:L_max
#             for A=1:length(att_strats)
#                 for j=1:length(att_strats[A])
#                     if value(c[r, i, A, j]) >= 0.5
#                         println("   Found c[r, i, A, j]=1 with r,i,A,j = $r, $i, $A, $j")
#                     end
#                 end
#             end
#         end
#     end

    if printing == 2
        # print all binary variables that are 1
        # @variable(LP_DO, s[r=1:length(v_0r), i=1:L_max, v=nodes], Bin)  # Ind. for d_r at node v at time i (in d_r's schedule)
        # @variable(LP_DO, w[r=1:length(v_0r), i=1:L_max, v=nodes_1, u=nodes_1], Bin)  # Ind. if both s_{i,v}^r and s_{i+1,u}^r are true (so if u-v is the path taken by d_r between (their) state i and i+1)
        # @variable(LP_DO, a[r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])], Bin)  # Ind. for if attacker arrives at their node j after d_r arrives at i (so t_i < t_j aka t_i - t_j < 0)
        # @variable(LP_DO, b[r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])], Bin)  # Ind. for if attacker arrives at their node j before d_r arrives at i
        # @variable(LP_DO, c[r=1:length(v_0r), i=1:L_max, A=1:length(att_strats), j=1:length(att_strats[A])], Bin)  # Ind. for if interdiction occurs at attacker node j (using strat A), defender resource r at node i (in their schedule)
        # @variable(LP_DO, z[A=1:length(att_strats)], Bin)

        # z variables
        for A=1:length(att_strats)
            if value(z[A]) > 0.5
                println("z[$A] is 1")
            end
        end

        # c variables
        for r=1:length(v_0r)
            for i=1:L_max
                for A=1:length(att_strats)
                    for j=1:length(att_strats[A])
                        if value(c[r, i, A, j]) > 0.5
                            println("c[$r, $i, $A, $j] is 1")
                        end
                    end
                end
            end
        end

        # b variables
        for r=1:length(v_0r)
            for i=1:L_max
                for A=1:length(att_strats)
                    for j=1:length(att_strats[A])
                        if value(b[r, i, A, j]) > 0.5
                            println("b[$r, $i, $A, $j] is 1")
                        end
                    end
                end
            end
        end
        
        # a variables
        for r=1:length(v_0r)
            for i=1:L_max
                for A=1:length(att_strats)
                    for j=1:length(att_strats[A])
                        if value(a[r, i, A, j]) > 0.5
                            println("a[$r, $i, $A, $j] is 1")
                        end
                    end
                end
            end
        end

        # w variables [r=1:length(v_0r), i=1:L_max, v=nodes_1, u=nodes_1
        for r=1:length(v_0r)
            for i=1:L_max
                for v=nodes_1
                    for u=nodes_1
                        if value(w[r, i, v, u]) > 0.5
                            println("w[$r, $i, $v, $u] is 1")
                        end
                    end
                end
            end
        end

        # s variables
        for r=1:length(v_0r)
            for i=1:L_max
                for v=nodes
                    if value(s[r, i, v]) > 0.5
                        println("s[$r, $i, $v] is 1")
                    end
                end
            end
        end
        
    end
    
    # return new_def_strat, obj_value, LP_DO

    return (
        new_def_strat=new_def_strat,
        obj_value=obj_value,
        LP_DO=LP_DO        
    )
end
# -

function convert_att_strat_to_non_backtracking(eig::EIG, att_strat)
    """
    Returns an attacker strategy dominating att_strat where no backtracking occurs by removing 'loops' and travelling slower on appropriate arcs

    Not in-place
    """
    new_att_strat_no_b = []  # new att strat where there is no backtracking
    visited_nodes = []
    
    successor = Dict(state => att_strat[i+1] for (i, state) in enumerate(att_strat[1:end-1]))  # the 'next' state for each state, will be mutated; (a linked list)
    most_recent_state_with_node = Dict()  # key is a node v in visited_nodes, value is the state (v, t) earliest in att_strat

    for state in att_strat
        if state.v in visited_nodes
            orig_v_state = most_recent_state_with_node[state.v]
            # delete visited nodes between orig_v_state and state; do it by truncating visited_nodes at the index where state.v is
            v_idx = findfirst(x -> x == state.v, visited_nodes)
            visited_nodes = visited_nodes[1:v_idx]
            
            # move around successor; A -> ... -> A -> B  ~~~> A -> B
            successor[orig_v_state] = successor[state]
        else
            most_recent_state_with_node[state.v] = state
            push!(visited_nodes, state.v)
        end
    end

    # no more loops, simply traverse successor and push states
    state_temp = att_strat[1]
    while state_temp.v != eig.network.v_∞
        push!(new_att_strat_no_b, state_temp)
        state_temp = successor[state_temp]
    end

    push!(new_att_strat_no_b, state_temp)  # push final state

    return new_att_strat_no_b
end

# +
function attackeroraclezhang(eig::EIG, def_strats, def_probs; L_max=nothing, optimiser=Gurobi.Optimizer, timeout=60.0, silent=true, printing=0, no_backtrack_strats=true)
    """
    Given available defender strategies, finds an optimal pure attacker strategy.
    Assumes integral data.
    
    def_strats: list of available defender strategies
    def_probs: list of probabilities of each defender strategy (given by coreLP)

    no_backtrack_strats: If true then, at the end, converts the new attacker strategy to the dominating one where no backtracking occurs by removing 'loops'
    
    nodes, edges, v_0, v_0r, exit_nodes, T_MAX, DELTA
    """
    if isnothing(L_max)
        L_max = eig.network.num_nodes  # maximum number of needed attacker states (note an attacker should never return to a previously visited node)
    end
    
    T_MAX = eig.t_max
    DELTA = eig.δ
    EPS_ = 0.25# * min(eig.δ, 1)  # for constraints (29), (30); assuming integral data
    
    bigM = T_MAX + 1  # should be enough for the big M constraints (29 & 30); T_MAX + 0.25 sometimes makes the LP infeasible
    
    nodes = 1:eig.network.num_nodes
    
    v_inf = eig.network.v_∞
    v_0 = eig.network.v_0
    v_0r = eig.network.v_0r
    edges = eig.network.edges

    LP_AO = Model(optimiser);
    if silent; set_silent(LP_AO); end
    # we use variable names (i, j) as given in our paper (not Zhang), except for s===A (our paper 's' is identified with variable name 'A')
    
    @variable(LP_AO, A[i=1:L_max, v=nodes], Bin)  # ind. for if attacker is at node v in state i
    @variable(LP_AO, w[i=1:L_max, e=1:length(edges)], Bin)  # ?? Restrict to 1:L_max-1 ()
    # @variable(LP_AO, a[S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:L_max], Bin)
    @variable(LP_AO, a[S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:length(def_strats[S][r])], Bin)
    # @variable(LP_AO, b[S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:L_max], Bin)
    @variable(LP_AO, b[S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:length(def_strats[S][r])], Bin)
    @variable(LP_AO, z[S=1:length(def_strats)], Bin)
    @variable(LP_AO, 0 ≤ t_a[i=1:L_max] ≤ T_MAX)
    
    @constraint(LP_AO, con22a, A[1, v_0] == 1)
    @constraint(LP_AO, con22b, A[L_max, v_inf] == 1)
    # @constraint(LP_AO, con22b[d=exit_nodes], A[L_max, d] == 1)  # slightly different to paper (due to excluding v_∞ from explicit graph)
    @constraint(LP_AO, con22c[i=1:L_max], sum( A[i, v] for v=nodes ) == 1)
    # @constraint(LP_AO, con23[i=1:L_max-1], A[i+1, v_inf] ≥ A[i, v_inf])
    # @constraint(LP_AO, con23[j=1:L_max, d=exit_nodes], A[j+1, d] ≥ A[j, d])  # slightly different to paper (due to excluding v_∞ from explicit graph)
    @constraint(LP_AO, con24[v=nodes, i=1:L_max-1], sum( A[i+1, u] for u in eig.neighbourhoods[v]) ≥ A[i, v])  # !! do dummy loop
    @constraint(LP_AO, con25a[e=1:length(edges), i=1:L_max-1], w[i, e] ≤ A[i, edges[e].a])
    @constraint(LP_AO, con25b[e=1:length(edges), i=1:L_max-1], w[i, e] ≤ A[i+1, edges[e].b])
    @constraint(LP_AO, con26[e=1:length(edges), i=1:L_max-1], w[i, e] ≥ A[i, edges[e].a] + A[i+1, edges[e].b] - 1)
    @constraint(LP_AO, con27, t_a[1] == 0)
    @constraint(LP_AO, con28[i=1:L_max-1], t_a[i+1] ≥ t_a[i] + sum( edges[e].dist * w[i, e] / eig.speed_A for e=1:length(edges)))
    
    # OLD
    # @constraint(LP_AO, con29a[S=1:length(def_strats), r=1:length(v_0r), i=1:length(def_strats[S][r]), j=1:L_max], - bigM * a[S, r, i, j] ≤ def_strats[S][r][i].t_a - t_a[j])
    # @constraint(LP_AO, con29b[S=1:length(def_strats), r=1:length(v_0r), i=1:length(def_strats[S][r]), j=1:L_max], def_strats[S][r][i].t_a - t_a[j] ≤ bigM * (1 - a[S, r, i, j]))
    # </OLD>
    
    # new (29) constraint
    @constraint(LP_AO, con29a[S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:length(def_strats[S][r])], - bigM * a[S, r, i, j] ≤ def_strats[S][r][j].t_a - t_a[i] - EPS_)  # CORRECT CONSTRAINT
    # @constraint(LP_AO, con29aW[S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:length(def_strats[S][r])], - bigM * a[S, r, i, j] ≤ def_strats[S][r][j].t_a - t_a[i])  # WRONG CONSTRAINT (just for experiments)
    @constraint(LP_AO, con29b[S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:length(def_strats[S][r])], def_strats[S][r][j].t_a - t_a[i] ≤ bigM * (1 - a[S, r, i, j]))
    
    # new (30) constraint
    @constraint(LP_AO, con30a[S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:length(def_strats[S][r])], - bigM * b[S, r, i, j] ≤ t_a[i] - def_strats[S][r][j].t_b - EPS_)  # CORRECT CONSTRAINT
    # @constraint(LP_AO, con30aW[S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:length(def_strats[S][r])], - bigM * b[S, r, i, j] ≤ t_a[i] - def_strats[S][r][j].t_b)  # WRONG CONSTRAINT (just for experiments)
    @constraint(LP_AO, con30b[S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:length(def_strats[S][r])], t_a[i] - def_strats[S][r][j].t_b ≤ bigM * (1 - b[S, r, i, j]))
    
    @constraint(LP_AO, con31[S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:length(def_strats[S][r])], z[S] ≥ a[S, r, i, j] + b[S, r, i, j] + A[i, def_strats[S][r][j].v] - 2)
    
    # @objective(LP_AO, Max, sum( (1 - z[S])*def_probs[S] for S=1:length(def_strats) ))  # original objective in Zhang paper
    @objective(LP_AO, Min, sum( z[S]*def_probs[S] for S=1:length(def_strats) ))  # directly outputs prob. of interdiction
    
    set_time_limit_sec(LP_AO, timeout)
    
    optimize!(LP_AO)
    
    if termination_status(LP_AO) == TIME_LIMIT
        if primal_status(LP_AO) != FEASIBLE_POINT
            println("Zhang AO timedout after ", timeout, " secs, but didn't find a feasible point")
            flush(stdout)
            # do something here? Like give it a bit more time to find a feasible point?
            return (
                new_att_strat=nothing,
                obj_value=nothing,
                LP_AO=LP_AO
            )
            # return nothing, nothing, LP_AO  # return essentially nothing
        elseif primal_status(LP_AO) == FEASIBLE_POINT
            println("Zhang AO timedout after ", timeout, " secs, found a feasible point")
            flush(stdout)
        end
    else
        # no timeout
        if termination_status(LP_AO) != OPTIMAL
            error("Zhang AO did not solve to optimality and did not timeout. ", termination_status(LP_AO))
        end
    end
    
    # extract attacker strategy from A variables
    new_att_strat = []
    for j=1:L_max
        # find vertex v such that A[j, v] == 1
        vertex = -1
        for v=nodes
            if value(A[j, v]) > 0.5
                vertex = v
                break;
            end
        end
        push!(new_att_strat, (v=vertex, t=value(t_a[j])))

        if vertex == v_inf
            break;  # already at v_inf so no need for anymore states
        end
    end

    if no_backtrack_strats
        new_att_strat = convert_att_strat_to_non_backtracking(eig, new_att_strat)
    end
    
    obj_value = objective_value(LP_AO)
    
    # TESTING
    
#     # interdict test
#     intdict = [doesinterdict(new_att_strat, def_strats[d], printing=true) for d=1:length(def_strats)]
    
#     for d=1:length(def_strats)
#         println("z value for d=$d is $(value(z[d]))")
        
#         if (value(z[d]) >= 0.5) && intdict[d] == 0 && def_probs[d] > 0
#             # z variable says interdiction occurs against attacker strat A, but doesinterdict says no interdiction
#             println("\n?????? For def_strat No. $d, z variable is 1, but doesinterdict says interdiction OCCURS NOT")
            
#             if def_probs[d] == 0
#                 println("Nevermind, the defender plays this strategy with prob 0")
#             end
#         end
#         if (value(z[d]) <= 0.5) && intdict[d] == 1 && def_probs[d] > 0
#             println("\n?????? For def_strat No. $d, z variable is 0, but doesinterdict says interdiction OCCURS")
#         end
#     end
    
#     # print out z = 1 variables (S)
#     for S=1:length(def_strats)
#         if value(z[S]) >= 0.5
#             println("   Found z[S]=1 with S = $S")
#         end
#     end
    
#     println()
    
#     # print out w = 1 variables (j, e)
#     for j in 1:L_max
#         for e in 1:length(edges)
#             if value(w[j, e]) >= 0.5
#                 println("   Found w[j,e]=1 with j,e = $j, $e")
#             end
#         end
#     end
    
#     println()
    
#     # print out A = 1 variables j=1:L_max, v=nodes
#     for j in 1:L_max
#         for v in nodes
#             if value(A[j, v]) >= 0.5
#                 println("   Found A[j, v]=1 with j,v = $j, $v")
#             end
#         end
#     end

#     println()
    
#     # print out a = 1 variables S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:L_max
#     for S=1:length(def_strats)
#         for r=1:length(v_0r)
#             for i=1:L_max
#                 for j=1:L_max
#                     if value(a[S, r, i, j]) >= 0.5
#                         println("   Found a[S, r, i, j]=1 with S,r,i,j = $S, $r, $i, $j")
#                     end
#                 end
#             end
#         end
#     end
    
#     println()
    
#     # print out b = 1 variables S=1:length(def_strats), r=1:length(v_0r), i=1:L_max, j=1:L_max
#     for S=1:length(def_strats)
#         for r=1:length(v_0r)
#             for i=1:L_max
#                 for j=1:L_max
#                     if value(b[S, r, i, j]) >= 0.5
#                         println("   Found b[S, r, i, j]=1 with S,r,i,j = $S, $r, $i, $j")
#                     end
#                 end
#             end
#         end
#     end

#     for r in 1:length(v_0r)
#         for i=1:L_max
#             for A=1:length(att_strats)
#                 for j=1:length(att_strats[A])
#                     if value(c[r, i, A, j]) >= 0.5
#                         println("   Found c[r, i, A, j]=1 with r,i,A,j = $r, $i, $A, $j")
#                     end
#                 end
#             end
#         end
#     end
    
    
    # return new_att_strat, obj_value, LP_AO

    return (
        new_att_strat=new_att_strat,
        obj_value=obj_value,
        LP_AO=LP_AO
    )
end

# +
# # get support of resulting mixed strategies
# att_supp = [a for a=1:length(results.att_probs) if results.att_probs[a] > 0]
# def_supp = [d for d=1:length(results.def_probs) if results.def_probs[d] > 0]

# # plot support scenarios
# for a in att_supp
#     for d in def_supp
#         animateEIGPure(the_eig, results.att_strats[a], results.def_strats[d], anim_time=5, fps=10, savefilename="att_strt $a vs def_strt $d", width=WIDTH)
#     end
# end

# # animateEIGPure(the_eig, results.att_strats[3], results.def_strats[4], anim_time=5, fps=10)

# +
# function defenderoraclezhangheuristic(att_strats)
#     """
#     Given available attacker strategies, finds an optimal pure defender strategy heuristically (using the betterDO in Zhang paper).
    
#     att_strats: list of available attacker strategies
#     """
#     error("NOT IMPLEMENTED")
# end

# +
# function attackeroraclezhangheuristic(def_strats)
#     """
#     Given available defender strategies, finds an optimal pure attacker strategy heuristically (using the betterAO in Zhang paper).
    
#     def_strats: list of available defender strategies
#     """
#     error("NOT IMPLEMENTED")
# end
# -

# #### New Oracles

# ##### Network Formulation

# +
function half_int_ceil(num)
    """
    Returns the smallest half_int larger than or equal to num
    """
    return ceil(num + 0.5) - 0.5
end

function half_int_floor(num)
    """
    Returns the largest half_int smaller than or equal to num
    """
    return floor(num - 0.5) + 0.5
end
# -

function att_strat_to_A_half_strat(dist_mtx, att_strat, speed_A; tol=1E-7)
    """
    Using the construction (Proposition 4.4) in our paper.
    """
    # find minimal j such that d(v_j, v_{j+1}) < t_{j+1} - t_j
    j = Inf
    for (i, state) in enumerate(att_strat[1:end-1])
        state_next = att_strat[i+1]
        if !isapprox(dist_mtx[state.v, state_next.v] / speed_A, state_next.t - state.t)  # if we code everything right, 
            # (assuming d(v_j, v_{j+1}) / s_A < t_{j+1} - t_j) is false) found j (minimal index with strict inequality)
            j = i
            break
        end
    end
    @assert(j != Inf, "should have found suitable j in converting att strat to half-int strat: att_strat=$att_strat")
    
    # create half strategy
    new_att_strat = []
    for (i, state) in enumerate(att_strat)
        if isapprox(state.t, round(state.t))
            # integer time, convert to half-int according to the formula
            new_t = i <= j ? round(state.t) + 0.5 : round(state.t) - 0.5
        else
            new_t = ceil(state.t) - 0.5
        end
        
        push!(new_att_strat, (v=state.v, t=new_t))
    end
    
    # new_att_strat = [(v=state.v, t=ceil(state.t) - 0.5) for state in att_strat]
    new_att_strat[1] = (v=att_strat[1].v, t=0.0)  # first state leaves at t=0
    return new_att_strat
end

function def_strat_to_t_max(eig::EIG, def_strat)
    """
    Returns def_strat, except with t_b in the last state extended to ceil(t_max)
    """
    for res in def_strat
        if res[end].t_b < eig.t_max
            # extend it
            new_t_b = ceil(eig.t_max)  # works assuming integral data (and \delta = 1)
            res[end] = (v = res[end].v, t_a = res[end].t_a, t_b = new_t_b)
        end
    end

    return def_strat
end

function constructattackeroraclegraph_SUPPBASED(eig::EIG; def_strats=[], res_just_left_times=nothing, printing=false)
    """   
    Constructs a partial attacker oracle graph based on the defender strats in def_strats (supposedly they are the support of the current defender strategy).
    Main idea is: Only construct arc type A1 (u, i, +)(v, j, -) if some strat s in def_strats covers v at time j-δ but not at time j.
    """
    # HALF_INTS = [i - 1/2 for i=1:ceil(eig.t_max)]  # does not contain 0 (though 0 gives some vertices in the time-extended network)
    
    nodes = 1:eig.network.num_nodes
    nodes_1 = 1:eig.network.num_nodes - 1  # all nodes except v_∞
    v_0 = eig.network.v_0
    v_inf = eig.network.v_∞
    
    @assert(nodes[end] == v_inf, "nodes[end] must be v_inf")

    # preprocess all node-time pairs (v, t) (t is half-int) where some strategy is covering (v, t)
    if isnothing(res_just_left_times)
        res_just_left_times = [Int[] for _ in nodes_1]  # vth entry is a list (sorted?) of times t where some strat in def_strats is covering (v, t - DELTA) but not (v, t)
        for def_strat in def_strats
            for res in def_strat
                for state in res
                    if state.v == v_0
                        continue  # attacker should never backtrack, so no need to add oppurtunistic arcs for v_0
                    end
    
                    # defender cannot be on v_inf, so no need to check
                    @assert(state.v != v_inf, "Defender cannot be on v_inf")
                    
                    if eig.dist_mtx[v_0, state.v] / eig.speed_A <= state.t_b < ceil(eig.t_max)  # no point in adding attacker arc with destination time >ceil(t_max) or one that is too fast
                        if !(state.t_b in res_just_left_times[state.v])  # we want unique elements in each list
                            push!(res_just_left_times[state.v], state.t_b)  # so the attacker may move to v at the half-int just after state.t_b
                        end
                    end
                end
            end
        end
    end

    for x in res_just_left_times
        unique!(x)  # ensure just-left-time for each node is seen exactly once
        sort!(x)  # maybe this is useful, maybe not?
    end
    
    # new_vs = [(v, i, b) for v in nodes for i in HALF_INTS for b in [-1, 1]]
    new_vs = []  # vertices (all of the form (v, i, b), where v is a node, i is 0 or a half-integer, b is -1 or 1)
    new_vs_pairless = []  # of the form (v, i) such that (v, i, +) and (v, i, -) are both in new_vs; DOESN'T contain (v_0, 0)
    active_vs = BinaryHeap(Base.By(last), [])  # min-heap of node-time pairs keyed by time still to be processed; (each (except for source and sink) will correspond to 2 vertices)
    # SYNTAX: (node_idx, time) where time is a half-int
    # add all oppurtunistic nodes to active_vs (so we don't have to check in the middle of for-loops below)
    for v in nodes_1
        if v == v_0
            continue
        end
        for t in res_just_left_times[v]
            push!(active_vs, (v, half_int_ceil(t)))
        end
    end
    
    # source and sink nodes
    source_node = (v_0, 0.0, 1)
    sink_node = (v_inf, ceil(eig.t_max) - 0.25, -1)  # (contents do not matter, as long as it is guaranteed not to be a key of an internal node 
    
    # arcs, represented as an adjacency list for each vertex
    arc_lists_o = Dict()  # arcs going out from each vertex (don't initialise with objects, do it uniformly inside the for-loops below)
    # initialise arc_lists_i with empty lists for all vertices (v, t) that are 'oppurtunistic' (i.e. have some def strat covering (v, t - δ) but not (v, t)), since we know these vertices must be processed
    arc_lists_i = Dict((v, half_int_ceil(t), -1) => [] for v in nodes_1 for t in res_just_left_times[v])  # arcs going into each vertex
    arc_lists_i[sink_node] = []  # initialise for sink node
    
    # process (v_0, 0)
    arc_lists_o[source_node] = []  # add to dictionary o
    for u in eig.neighbourhoods[v_0]
        travel_time = eig.dist_mtx[v_0, u] / eig.speed_A  # an integer, assuming appropriately scaled integral data
        t = half_int_ceil(travel_time)  # round up to nearest half-int (must travel slightly slower, cannot travel faster than max-speed - note our theory guarantees no loss of optimality)

        if u == v_inf
            # very weird (v_0-v_inf arc existing means attacker wins trivially)
            println("V_0-V_INF ARC DETECTED")  # really, this should be detected in code much sooner before running any oracles
            if eig.dist_mtx[v_0, v_inf] < eig.t_max
                push!(arc_lists_o[source_node], sink_node)
            end
            continue
        end
        
        flag_max_speed_is_opp = false  # true if it is found that the max-speed arc is also an oppurtunistic flag
        
        # add arcs at oppurtunistic times (times just after some resource leaves)
        for t_left in res_just_left_times[u]
            t_left = half_int_ceil(t_left)  # some resource has left u by this time
            if t_left < t
                continue  # attacker cannot travel arc (too quick)
            elseif t_left + eig.dist_mtx[u, v_inf] / eig.speed_A > eig.t_max
                continue  # attacker cannot reach v_inf in time 
            end
            
            if t_left == t
                flag_max_speed_is_opp = true
            end
            
            # add oppurtunistic arc (v_0, 0, 1)(u, t_left, -1)
            new_v = (u, t_left, -1)
            push!(arc_lists_o[source_node], new_v)
            push!(arc_lists_i[new_v], source_node)

            if printing
                println("  !! Added opp arc $source_node -> $new_v")
            end
            
            # oppurtunistic node new_v is already in active_vs by preprocessing
        end

        if !flag_max_speed_is_opp
            # add max-speed arc (v_0, 0, 1)(u, t, -1) 
            new_v = (u, t, -1)  # note t is half-int
            push!(arc_lists_o[source_node], new_v)
            arc_lists_i[new_v] = [source_node]
    
            push!(active_vs, new_v)  # non-oppurtunistic, must add to heap active_vs
        end
    end

    if printing
        println("Done processing (v_0, 0) (v_0=$v_0). outgoing and incoming arcs are resp.:")
        display(arc_lists_o)
        display(arc_lists_i)
    end
    
    
    # process rest of nodes
    while !isempty(active_vs)
        node_time_pair = pop!(active_vs)        
        v = node_time_pair[1]
        t = node_time_pair[2]  # should be a half-int (if v isn't v_0)
        @assert(isinteger(t - 1/2), "time $t should be a half-int (node $v in active_vs)")
        @assert(v != v_inf, "v_inf shouldn't be in active_vs")
        
        if (v, t, 1) in new_vs
            if printing
                println("!!!!!!!! SKIPPING ALREADY PROCESSED NODE ($v, $t)")
            end
            @assert(false, "could have been a duplicate node without the check (v, t, 1) in new_vs. (v, t, 1) = ($v, $t, 1)")
            continue  # already processed (is there a better way that avoids this?)
        end
        
        # put appropriate nodes into new_vs
        push!(new_vs, (v, t, -1))
        curr_v = (v, t, 1)  # node from which we will now draw arcs from
        push!(new_vs, curr_v)

        push!(new_vs_pairless, (v, t))
        
        arc_lists_o[curr_v] = []  # add to dictionary o
        
        for u in eig.neighbourhoods[v]
            if u == v_0
                continue  # attacker should never backtrack
            end
            
            travel_time = eig.dist_mtx[v, u] / eig.speed_A  # an integer
            @assert(isinteger(travel_time), "travel_time $travel_time should be an integer (node $v to node $u)")
            
            if t + travel_time + eig.dist_mtx[u, v_inf] / eig.speed_A > eig.t_max
                continue  # attacker shouldn't go to (u, t + travel_time) as there is not enough time to make it to v_inf
            end

            if u == v_inf
                # link straight to sink (we checked above that it is feasible)
                push!(arc_lists_o[curr_v], sink_node)
                push!(arc_lists_i[sink_node], curr_v)
                continue
            end
            
            flag_max_speed_is_opp = false  # true if it is found that the max-speed arc is also an oppurtunistic flag
            
            # add oppurtunistic arcs
            for t_left in res_just_left_times[u]
                t_left = half_int_ceil(t_left)
                if t_left < t + travel_time
                    continue  # attacker cannot travel arc (too quick)
                elseif t_left + eig.dist_mtx[u, v_inf] / eig.speed_A > eig.t_max
                    continue  # attacker cannot reach v_inf in time 
                end

                if t_left == t + travel_time
                    flag_max_speed_is_opp = true
                end
                
                # add oppurtunistic arc (v_0, 0, 1)(u, t_left, -1)
                new_v = (u, t_left, -1)
                push!(arc_lists_o[curr_v], new_v)
                push!(arc_lists_i[new_v], curr_v)
                
                if printing
                    println("  !! Added opp arc $curr_v -> $new_v")
                end
                
                # oppurtunistic node new_v is already in active_vs by preprocessing
            end

            if !flag_max_speed_is_opp
                # add max-speed arc (v, t, 1)(u, t + d(v, u) / s_A, -1)
                new_v = (u, t + travel_time, -1)
                push!(arc_lists_o[curr_v], new_v)

                if printing
                    temp = haskey(arc_lists_i, new_v)
                    println("adding non-opp max-speed arc ($v, $t, 1)-($u, $(t + travel_time), -1); haskey(arc_lists_i, new_v) == $temp")
                end
                
                if !haskey(arc_lists_i, new_v)  # maybe there is a smart way to avoid having to do this check?
                    arc_lists_i[new_v] = []
                    push!(active_vs, new_v)  # add to heap to be processed WARNING: POSSIBLE DUPLICATES?
                end
                
                push!(arc_lists_i[new_v], curr_v)
    
                # we know new_v is not in new_vs since t + travel_time > t and curr_v has minimal t
            else
                if printing
                    println("    !! max-speed arc ($v, $t, 1)-($u, $(t + travel_time), -1) was opp")
                end
            end
        end
    end
    
    push!(new_vs, source_node)  # note source and sink are last in new_vs
    push!(new_vs, sink_node)
    
    # add A2 arcs for nodes in new_vs; use new_vs_pairless so that each (v, t) +-pair is processed exactly once
    for (v, t) in new_vs_pairless
        arc_lists_o[(v, t, -1)] = [(v, t, 1)]  # -1 node only has 1 outgoing arc to its +1 counterpart
        arc_lists_i[(v, t, 1)] = [(v, t, -1)]  # +1 node only has 1 incoming arc from its -1 counterpart
    end

    # give full neighbourhoods for source and sink (so that formatting in arc_lists_i and arc_lists_o is consistent)
    arc_lists_i[source_node] = []
    arc_lists_o[sink_node] = []
    
    # A3 arcs are now captured by sending any virtual exit node directly to sink (which has destination time t_max) if feasible

    sort!(new_vs_pairless, by = x -> x[2])  # maybe useful, maybe not
    
    return (new_vs=new_vs, arc_lists_o=arc_lists_o, arc_lists_i=arc_lists_i, new_vs_pairless=new_vs_pairless, source_node=source_node, sink_node=sink_node)
end

function does_def_strat_cover_vt(def_strat, v, t)
    """
    Simply function that returns true if def_strat covers (original) node v at time t, false otherwise.

    def_strat: defender strategy
    v: node index (of original EIG)
    t: time
    """
    for res in def_strat
        for state in res
            if state.v != v
                continue
            end

            if state.t_a <= t <= state.t_b
                return true
            end
        end
    end

    return false
end

# +
function attackeroraclenew(eig::EIG, def_strats, def_probs; supp_tol=1e-7, optimiser=Gurobi.Optimizer, timeout=60.0, silent=true, printing=false, no_backtrack_strats=true)
    """
    Given available defender strategies, finds an optimal pure attacker strategy.
    
    def_strats: list of available defender strategies
    def_probs: list of probabilities resp. for def_strats
    supp_tol: tolerance for determining whether a def strat is in the support (prob > supp_tol) or not (prob <= supp_tol)
    no_backtrack_strats: If true then, at the end, converts the new attacker strategy to the dominating one where no backtracking occurs by removing 'loops'
    """
    def_supp = []  # def strats in the current support 
    def_supp_probs = []  # def strats in the current support

    for (idx, s) in enumerate(def_strats)
        if def_probs[idx] > supp_tol
            push!(def_supp, s)
            push!(def_supp_probs, def_probs[idx])
        end
    end

    # construct graph
    time_new_graph_construction = @elapsed new_graph = constructattackeroraclegraph_SUPPBASED(eig, def_strats=def_supp, printing=printing)
    num_nodes_new_graph = length(new_graph.new_vs)
    num_edges_new_graph = sum(length(new_graph.arc_lists_o[new_v]) for new_v in new_graph.new_vs)
    
    # HALF_INTS = [i - 1/2 for i=1:ceil(eig.t_max)]  # does not contain 0 (though 0 gives some vertices in the time-extended network)
    
    # nodes = 1:eig.network.num_nodes
    # nodes_1 = 1:eig.network.num_nodes - 1
    v_0 = eig.network.v_0
    v_inf = eig.network.v_∞
    
    new_vs = new_graph.new_vs  # vertices (all of the form (v, i, b), where v is a node, i is 0 or a half-integer, b is -1 or 1)
    
    # arcs, represented as an adjacency list for each vertex
    arc_lists_o = new_graph.arc_lists_o  # arcs going out from each vertex
    arc_lists_i = new_graph.arc_lists_i  # arcs going into each vertex
    
    # source and sink nodes (for brevity)
    source_node = new_graph.source_node #(v_0, 0, 1)
    sink_node = new_graph.sink_node #(v_inf, ceil(eig.t_max) - 0.25, -1)

    @assert(new_vs[end-1] == source_node && new_vs[end] == sink_node, "Final two entries of new_vs must be [source_node, sink_node]")
    
    # form MILP
    LP_AOn = Model(optimiser);
    if silent; set_silent(LP_AOn); end
    
    @variable(LP_AOn, x[a=new_vs, b=new_vs; b in arc_lists_o[a]], Bin)  # arc flow variables
    @variable(LP_AOn, z[y=1:length(def_supp)], Bin)  # indicator variable for if def strat y interdicts att strat given by flows
    
    @constraint(LP_AOn, con3p1[a=new_vs[1:end-2]], sum( x[b, a] for b in arc_lists_i[a] ) == sum( x[a, c] for c in arc_lists_o[a] ) )  # flow balance at each internal node: NOTE need source and sink to be final 2 in new_vs
    @constraint(LP_AOn, con3p2, sum( x[source_node, b] for b in arc_lists_o[source_node] ) == 1)
    @constraint(LP_AOn, con3p3, sum( x[a, sink_node] for a in arc_lists_i[sink_node] ) == 1)

    # 2nd approach, iterate (v, t) through new_vs_pairless and add appropriate con3p4 constraints (after defining variable in ILP)
    for (v, t) in new_graph.new_vs_pairless
        # find def strats that cover (v, t)
        for (def_strat_idx, def_strat) in enumerate(def_supp)
            if does_def_strat_cover_vt(def_strat, v, t)
                # include constraint for this def_strat, v, t combination
                @constraint(LP_AOn, z[def_strat_idx] >= x[(v, t, -1), (v, t, 1)])
            end
        end
    end
    
    # @objective(LP_AOn, Max, sum( (1 - def_probs[y]) * z[y] for y=1:length(def_strats) ))
    @objective(LP_AOn, Min, sum( def_supp_probs[y] * z[y] for y=1:length(def_supp) ))  # prob of interdiction   
    
    set_time_limit_sec(LP_AOn, timeout)
    
    optimize!(LP_AOn)
    
    if termination_status(LP_AOn) == TIME_LIMIT
        if primal_status(LP_AOn) != FEASIBLE_POINT
            println("Network AO timedout after ", timeout, " secs, but didn't find a feasible point")
            flush(stdout)
            # do something here? Like give it a bit more time to find a feasible point?
            return (new_att_strat=nothing,
                obj_value=nothing,
                LP_AO=LP_AOn,
                num_nodes_new_graph=num_nodes_new_graph,
                num_edges_new_graph=num_edges_new_graph,
                time_new_graph_construction=time_new_graph_construction)
            # return nothing, nothing, LP_AOn  # return essentially nothing
        elseif primal_status(LP_AOn) == FEASIBLE_POINT
            println("Network AO timedout after ", timeout, " secs, found a feasible point")
            flush(stdout)
        end
    else
        # no timeout
        if termination_status(LP_AOn) != OPTIMAL
            # compute conflict set
            compute_conflict!(LP_AOn)
            if get_attribute(LP_AOn, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
               iis_model, _ = copy_conflict(LP_AOn)
               print(iis_model)
            end
            
            error("Network AO did not solve to optimality and did not timeout. ", termination_status(LP_AOn), primal_status(LP_AOn))
        end
    end
    
#     for a in new_vs
#         if a == sink_node
#             println("CHECKING THE SINK")
#         end
        
#         for b in arc_lists_o[a]
#             if value(x[a, b]) > 0
#                 yeet = value(x[a, b])
#                 println("Found x[$a, $b] = $yeet")
#             end
#         end
        
#         for b in arc_lists_i[a]
#             if value(x[b, a]) > 0
#                 yeet = value(x[b, a])
#                 println("Found x[$b, $a] = $yeet")
#             end
#         end
        
#         if a == sink_node
#             println("FIN CHECKING THE SINK")
#         end
#     end
    
    # extract attacker strategies   
    # curr_flows = Dict((a, b) => value(x[a, b]) for a in new_graph.new_vs for b in new_graph.arc_lists_o[a])
    
    # while sum(curr_flows[(source_node, b)] for b in arc_lists_o[source_node]) > 0.5
        # find a path of non-zero flow
        # min_arc_flow = 1.0  # minimum flow encountered along path

    new_att_strat = [(v=v_0, t=0.0)]  # att strat; start at (v_0, 0)  (use float for type consistency)
    # curr_v = v_0  # vertex of original graph attacker is currently on
    # curr_time = 0
    curr_node = source_node  # current node in time-ext. graph

    while curr_node[1] != v_inf
        # find flow from (curr_node, curr_time, 1)
        for node in arc_lists_o[curr_node]
            if value(x[curr_node, node]) > 0.5
                # node is the next node in attacker flow
                if node[1] != curr_node[1]  # we only want to record (v, i) once
                    push!(new_att_strat, (v=node[1], t=node[2]))  # recall node is of the form (v, i, 1) or (v, i, -1); however, we only want to record (v, i) once
                end

                curr_node = node

                break;  # next half_int
            end
        end
        
        # display(curr_node)
    end

    if no_backtrack_strats
        new_att_strat = convert_att_strat_to_non_backtracking(eig, new_att_strat)
    end
    
    obj_value = objective_value(LP_AOn)
    
    return (new_att_strat=new_att_strat,
        obj_value=obj_value,
        LP_AO=LP_AOn,
        num_nodes_new_graph=num_nodes_new_graph,
        num_edges_new_graph=num_edges_new_graph,
        time_new_graph_construction=time_new_graph_construction)
end
# -

function constructdefenderoraclegraph_SUPPBASED(eig::EIG; att_strats=[], att_cover_times=nothing, printing=false)
    """
    Constructs a partial defender oracle graph based on the attacker strats in def_strats (supposedly they are the support of the current attacker strategy).
    Main idea is: Only construct arc type A1 (u, i, +)(v, j, -) if some strat s in def_strats covers v at time j-δ but not at time j.
    """
    nodes = 1:eig.network.num_nodes
    nodes_1 = 1:eig.network.num_nodes - 1  # nodes without v_inf
    v_0 = eig.network.v_0
    v_inf = eig.network.v_∞

    @assert(nodes[end] == v_inf, "nodes[end] must be v_inf")
    
    if isnothing(att_cover_times)
        # compute attacker cover times
        att_cover_times = [Int[] for _ in nodes_1]  # indexed by nodes (not v_inf) with vth entry the collection of times t - 0.5 where some att strat in att_strats is on v at t

        shortest_r_dist = [minimum(div(eig.dist_mtx[v_0r, v], eig.speed_D) for v_0r in eig.network.v_0r) for v in nodes_1]  # for truncating due to shortest defender paths
        
        for att_strat in att_strats
            for state in att_strat
                if state.t == 0 || state.v == v_inf
                    continue  # don't include start or end (defender can never interdict here)
                end

                if state.t < shortest_r_dist[state.v]
                    continue  # defender can never reach state.v at time state.t, so no point in trying to cover it
                end
                
                floor_time = state.t - 0.5
                @assert(isinteger(floor_time), "state.t - 0.5 = $floor_time should be an integer")
                @assert(state.t <= eig.t_max, "state.t = $state.t should be at most eig.t_max")
                push!(att_cover_times[state.v], Int(floor_time))
            end
        end
    end

    # sort each att_cover_times list (in preparation for avoiding double-pushing to new_vs)
    for x in att_cover_times
        unique!(x)  # ensure each cover time is used exactly once
        sort!(x)
    end
    
    # vertices (all of the form (v, i), where v is a node, i is 0 an integer (or special source/sink nodes)
    new_vs = []
    # new_vs_just_before_cover = []  # sub-list of new_vs of all (v, t) such that (v, t+0.5) is attacker-covered; will be iterated over in the main loop for drawing arcs
        # doesn't include source nor sink node
        # TODO: is this now useless with the is_new_vs_just_before_cover flag?
    
    # add R_0 and R_∞ (source and sink), keys don't matter (-1 in left to ensure representation is not in another node)
    source_node = (-1, -1)
    sink_node = (-1, Int(ceil(eig.t_max)) + 2)  # key doesn't matter (as long as time-coordinate is bigger than all others)
    
    push!(new_vs, source_node)
    
    # add v_0^r vertices
    for v_0r in unique(eig.network.v_0r)  # ensure each resource-initial node is only done once
        push!(new_vs, (v_0r, 0))
    end
    
    # add internal vertices and t_max vertices
    for v in nodes_1
        prev_t = -2  # t value in the previous iteration in the following for-loop
        
        for t in att_cover_times[v]
            # add vertices 'surrounding' the attacker cover point (recall t+0.5 is the time that the attacker actually arrives at v)            
            # only add (v, t) if it wasn't added last iteration (assuming att_cover_times[v] is sorted)
            if prev_t + 1 != t
                push!(new_vs, (v, t))
            end

            push!(new_vs, (v, t + 1))  # always add this node
            
            # push!(new_vs_just_before_cover, (v, t))  # only push (v, t)
            prev_t = t
        end
        
        # add t_max vertices (only if it wasn't added in the final iteration above, and either v is some v_0r or att_cover_times[v] is non-empty)
        if prev_t + 1 != ceil(eig.t_max)
            if !(isempty(att_cover_times[v])) || (v in eig.network.v_0r)
                push!(new_vs, (v, Int(ceil(eig.t_max))))
            end
        end
    end
    
    push!(new_vs, sink_node)
    
    sort!(new_vs, by = x -> x[2])  # sort new_vs by time-coordinate; NOTE: source_node is first, sink_node is last
    # sort!(new_vs_just_before_cover, by = x -> x[2])
    
    num_nodes_new_graph = length(new_vs)  # note we know no other vertex should be created (unlike the attacker network)
    if printing
        println("Constructed defender graph vertices: $num_nodes_new_graph")
        display(new_vs)
    end

    is_new_vs_just_before_cover = Dict(node => false for node in new_vs)  # indexed by new_vs (v, t); true if (v, t+0.5) is att-covered, false otherwise
    is_new_vs_just_after_cover = Dict(node => false for node in new_vs)  # indexed by new_vs (v, t); true if (v, t-0.5) is att-covered, false otherwise

    # determine is_new_vs_just_before_cover and is_new_vs_just_after_cover
    for v in nodes_1 
        for t in att_cover_times[v]
            # handle vertices 'surrounding' the attacker cover point (recall t+0.5 is the time that the attacker actually arrives at v)
            is_new_vs_just_before_cover[(v, t)] = true
            is_new_vs_just_after_cover[(v, t + 1)] = true
        end

        # note an attacker cover can only take place at some half-int between 0 and ceil(eig.t_max)
    end

    if printing
        println("is_new_vs_just_before_cover:")
        display(is_new_vs_just_before_cover)
        println("is_new_vs_just_after_cover:")
        display(is_new_vs_just_after_cover)
    end
    
    # draw edges
    arc_lists_i = Dict(node => [] for node in new_vs)  # incoming arcs, indexed by each new_vs
    arc_lists_o = Dict(node => [] for node in new_vs)  # outgoing arcs, indexed by each new_vs
    out_expecting_node_times = [Int[] for _ in nodes_1]  # indexed by nodes_1,
        # vth entry is the current list of times t in chronological order for which we have already computed all incoming edges for (v, t) and are expecting/ready
        # to draw an outgoing arc
        # Note we store the t value such that (v, t-0.5) is attacker-covered (since then (v, t) is the node for which we will draw outgoing arcs)
    most_recent_processed_node_times = [-1 for _ in nodes_1]  # indexed by nodes_1, gives the time t of the most recently processed (v, t) in new_vs
        # -1 by default (meaning no node pair (v, t) in new_vs has been processed yet)
    
    # draw source_node edges
    for v_0r in unique(eig.network.v_0r)  # ensure each resource-initial node is only done once
        start_v = (v_0r, 0)
        push!(arc_lists_o[source_node], start_v)
        push!(arc_lists_i[start_v], source_node)
        
        push!(out_expecting_node_times[v_0r], 0)  # we will draw arcs out of (v_0r, 0)
        most_recent_processed_node_times[v_0r] = 0
        
        if printing
            println("Added source arc to $start_v")
        end
    end
    
    # draw internal arcs and sink_node arcs; iterate through new_vs in chronological order
    for node_time_pair in new_vs[2:end-1]  # (v, t); avoid source and sink nodes (guaranteed to be first and last of new_vs if new_vs is sorted)
    # for node_time_pair in new_vs_just_before_cover  # (v, t)
        @assert(node_time_pair != source_node, "node_time_pair should not be source_node when drawing internal edges for def graph")
        @assert(node_time_pair != sink_node, "node_time_pair should not be sink_node when drawing internal edges for def graph")

        v = node_time_pair[1]
        t = node_time_pair[2]
        
        @assert(isinteger(t), "t = $t should be an integer")
        # @assert(t != 0, "t should not be 0 (an attacker should never be able to cover time 0.5)")

        if t == 0
            @assert(v in eig.network.v_0r)
            continue  # process resource-inital nodes separately (done above)
        end
        
        # add arc (v, t') -> (v, t) for latest t' such that (v, t') has already been processed
        if most_recent_processed_node_times[v] != -1
            tail_node = (v, most_recent_processed_node_times[v])
            push!(arc_lists_o[tail_node], node_time_pair)
            push!(arc_lists_i[node_time_pair], tail_node)

            if printing
                println("  Added waiting arc $tail_node -> $node_time_pair")
            end
        end
        
        if t == ceil(eig.t_max)  # WARNING: this will never trigger in the for-loop if using new_vs_just_before_cover
            # only draw the incoming 'waiting' edge from the latest time in in_complete_node_times[v] (done above) and the sink edges
            if most_recent_processed_node_times[v] != -1  # if -1, then v is not some v_0r and no attacker was covering
            # if !isempty(out_expecting_node_times[v])  # if empty, then v is not some v_0r and no attacker was covering: THIS IS WRONG, attacker could cover ceil(t_max) - 0.5
                # draw sink_node edges
                push!(arc_lists_o[node_time_pair], sink_node)
                push!(arc_lists_i[sink_node], node_time_pair)

                if printing
                    println("Added sink arc from $node_time_pair")
                end
            else
                if printing
                    println("    !!!! FOUND NON-V_0R NON-ATTACKER-COVERED VERTEX $v IN MAIN LOOP OF CONSTRUCTING DEFENDER ARCS")
                    println("    !!!! out_expecting_node_times[$v] = $(out_expecting_node_times[v])")
                end
            end

            # TODO: include t (=ceil(t_max)) in out_expecting_node_times?
            continue  # no other arcs
        end
        
        if is_new_vs_just_before_cover[node_time_pair]
            # draw incoming arcs to (v, t) (corresponds with shortest paths ending at v)
            for w in nodes_1
                if w == v
                    continue  # handle waiting arcs separately
                end
                
                if !isempty(out_expecting_node_times[w])
                    # there is some (w, t) that is ready to have an outgoing arc, so add some arc if defender can make it from w to v in time to cover (v, t + 0.5)
                    travel_time_wv = eig.dist_mtx[w, v] / eig.speed_D
    
                    flag_break = false  # true if we break out of the following for-loop, false otherwise
                    
                    for idx in length(out_expecting_node_times[w]) : -1 : 1  # iterate in reverse order to find the latest time that fits
                        time_w = out_expecting_node_times[w][idx]
                        if t < time_w + travel_time_wv
                            continue  # defender cannot make it from w to v in time to cover (v, t + 0.5)
                        end
                        
                        tail_node = (w, time_w)
                        if !is_new_vs_just_after_cover[tail_node] && time_w > 0
                            continue  # shouldn't draw outgoing arcs from nodes that aren't just after an attacker cover (unless it's a waiting arc or its from time 0)
                        end
                        
                        # t >= time_w + travel_time_wv, so draw arc (w, time_w) -> (v, t)
                        push!(arc_lists_o[tail_node], node_time_pair)
                        push!(arc_lists_i[node_time_pair], tail_node)
    
                        if printing
                            println("  Added arc $tail_node -> $node_time_pair; note out_expecting_node_times[$w] = $(out_expecting_node_times[w]) and travel_time_wv = $travel_time_wv")
                        end
                        flag_break = true
                        break  # only one arc per w
                    end
                    
                    if !flag_break
                        # if no arc was constructed, that is okay?
                        if printing
                            println("    !!!! No arc ($w, ?) -> $node_time_pair; note out_expecting_node_times[$w] = $(out_expecting_node_times[w]) and travel_time_wv = $travel_time_wv")
                        end
                    end
                end
            end
        end

        # # add arc (v, t) -> (v, t+1) only if it will cover some att strat - Actually, this is taken care of in the 'normal' waiting arcs??
        # @assert(t < ceil(eig.t_max), "t = $t should be < ceil(eig.t_max) = $(ceil(eig.t_max))")
        # head_node = (v, t + 1)
        # push!(arc_lists_o[node_time_pair], head_node)
        # push!(arc_lists_i[head_node], node_time_pair)
        
        if is_new_vs_just_after_cover[node_time_pair]
            push!(out_expecting_node_times[v], t)  # it makes sense to have an outgoing arc (to some (u, t') where u != v)
        end
        most_recent_processed_node_times[v] = t
    end
    
    # # draw arcs for t_max nodes
    # for v in nodes_1
    #     # only draw the incoming 'waiting' edge from the latest time in in_complete_node_times[v]
    #     if most_recent_processed_node_times[v] != -1  # if -1, then v is not some v_0r and no attacker was covering
    #         t_max_node = (v, ceil(eig.t_max))
            
    #         # add arc (v, t') -> (v, t) for latest t' such that (v, t') has already been processed
    #         tail_node = (v, most_recent_processed_node_times[v])
    #         push!(arc_lists_o[tail_node], t_max_node)
    #         push!(arc_lists_i[t_max_node], tail_node)
            
    #         # draw sink_node edges
    #         push!(arc_lists_o[t_max_node], sink_node)
    #         push!(arc_lists_i[sink_node], t_max_node)
    #         if printing
    #             println("Added t_max waiting arc $tail_node -> $t_max_node")
    #             println("Added sink arc from $t_max_node")
    #         end
    #     end
    # end

    # TODO: include t (=ceil(t_max)) in out_expecting_node_times for consistency - should we return out_expecting_node_times?
    num_edges = sum(length(arc_lists_o[new_v]) for new_v in new_vs)
    if printing
        println("Constructed defender edges: $num_edges edges")
    end
    
    return (new_vs=new_vs, arc_lists_o=arc_lists_o, arc_lists_i=arc_lists_i, source_node=source_node, sink_node=sink_node)
end

# +
function defenderoraclenew(eig::EIG, att_strats_, att_probs; supp_tol=1e-7, optimiser=Gurobi.Optimizer, timeout=60.0, silent=true, printing=false)
    """
    Given available attacker strategies, finds an optimal pure defender strategy.
    
    att_strats: list of available attacker strategies
    att_probs: list of probabilities resp. for att_strats
    supp_tol: tolerance for determining whether a att strat is in the support (prob > supp_tol) or not (prob <= supp_tol)
    """
    @assert(eig.δ == 1, "Delta should be 1 but it is $(eig.δ)")
    
    att_strats = [att_strat_to_A_half_strat(eig.dist_mtx, att_strat, eig.speed_A) for att_strat in att_strats_]  # convert all att_strats to A_{1/2} strat

    att_supp = []  # att strats in the current support 
    att_supp_probs = []  # att strat probs in the current support

    for (idx, s) in enumerate(att_strats)
        if att_probs[idx] > supp_tol
            push!(att_supp, s)
            push!(att_supp_probs, att_probs[idx])
        end
    end
    
    # construct graph (assuming DELTA == 1?)
    time_new_graph_construction = @elapsed new_graph = constructdefenderoraclegraph_SUPPBASED(eig, att_strats=att_supp, printing=printing)
    num_nodes_new_graph = length(new_graph.new_vs)
    num_edges_new_graph = sum(length(new_graph.arc_lists_o[new_v]) for new_v in new_graph.new_vs)
    
    # HALF_INTS = [i - 1/2 for i=1:Int(ceil(eig.t_max))]  # does not contain 0 (though 0 gives some vertices in the time-extended network)
    
    nodes = 1:eig.network.num_nodes
    nodes_1 = 1:eig.network.num_nodes - 1
    v_0 = eig.network.v_0
    v_inf = eig.network.v_∞
    
    new_vs = new_graph.new_vs  # vertices (all of the form (v, i), where v is a node, i is 0 or a positive integer)
    
    # arcs, represented as an adjacency list for each vertex
    arc_lists_o = new_graph.arc_lists_o  # arcs going out from each vertex
    arc_lists_i = new_graph.arc_lists_i  # arcs going into each vertex
    
    # source and sink nodes (for brevity)
    source_node = new_graph.source_node
    sink_node = new_graph.sink_node
    # sink_node = (v_inf, ceil(eig.t_max) + 0.5, -1)
    
    # # compute ind_xvi
    # # the report has 1_{x, v, i-1/2} = 1 iff att strat x is on node v (original graph) at time i-1/2, for integers i,
    # # we define ind_xvi[x, v, i] == 1_{x, v, i-1/2}, thus ind_xvi[x, v, i] = 1 if att strat x is on node v (original graph) at time i-1/2, for integers i,
    # ind_xvi = [0 for x in att_strats, v in nodes, i in HALF_INTS]  # assume 0 until we can find a counterexample
    
    # for (x_idx, x) in enumerate(att_strats)
    #     # process schedule and trigger appropriate indicators if a counterexample is found
    #     for state in x[2:end]  # don't check (v_0, 0) state
    #         i_h_idx = Int(state.t + 0.5)
    #         ind_xvi[x_idx, state.v, i_h_idx] = 1
    #     end
    # end

    # # finally, we define ind_xvi_full as the indicator for if att strat x is on node v at *any time* within [i - eig.delta, i]
    # ind_xvi_full = [0 for x in att_strats, v in nodes, i in 1:ceil(eig.t_max)]  # assume 0 until we can find a counterexample
    # for (x_idx, x) in enumerate(att_strats)
    #     # process schedule and trigger appropriate indicators if a counterexample is found
    #     for state in x[2:end]  # don't check (v_0, 0) state (defender will never be there by assumption)
    #         for i in floor(state.t):ceil(state.t) + eig.δ  # note state.t is a half-int
    #             if i - eig.δ <= state.t <= i <= ceil(eig.t_max)
    #                 ind_xvi_full[x_idx, state.v, Int(i)] = 1  # state.t in [i - eig.delta, i]
    #                 # println("???????????? ind_xvi_full[$x_idx, $(state.v), $i] is 1")
    #             end
    #         end
    #     end
    # end
    
    # form MILP
    LP_DOn = Model(optimiser);
    if silent; set_silent(LP_DOn); end
    
    @variable(LP_DOn, x[a=new_vs, b=new_vs; b in arc_lists_o[a]] ≥ 0, Int)  # arc flow variables
    @variable(LP_DOn, z[y=1:length(att_supp)], Bin)  # indicator variable for if att strat y interdicts def strat given by flows
    
    @assert(new_vs[1] == source_node && new_vs[end] == sink_node, "new_vs[1] must be source_node, and new_vs[end] must be sink_node")
    
    @constraint(LP_DOn, con1[a=new_vs[2:end-1]], sum( x[b, a] for b in arc_lists_i[a] ) == sum( x[a, c] for c in arc_lists_o[a] ) )  # flow balance at each internal node
    
    # compute c_r := c_{v_0^r} values
    c_r = counter(eig.network.v_0r)
    unique_rs = unique(eig.network.v_0r)
    
    @constraint(LP_DOn, con2[r=unique_rs], x[source_node, (r, 0)] == c_r[r] )
    @constraint(LP_DOn, con3, sum( x[a, sink_node] for a in arc_lists_i[sink_node] ) == length(eig.network.v_0r))

    # APPROACH: manually find the set of all y[(v, i), (v, i + 1)] that go into the sum for the constaint z[y]
    new_vs_just_before_cover = [[] for _ in 1:length(att_supp)]  # indexed by att_supp y, the list of (v, i) such that y covers (v, i+0.5)
    for (att_idx, att_strat) in enumerate(att_supp)
        # iterate through att_strat, manually adding
        for state in att_strat[2:end-1]  # skip initial state and end state
            @assert(state.v != v_inf, "internal state of att strat should not have v_inf: att_strat: $att_strat")
            
            vi = (state.v, Int(state.t - 0.5))

            # ensure vi is actually in new_vs (i.e. check distance condition)
            shortest_r_dist = minimum(div(eig.dist_mtx[v_0r, state.v], eig.speed_D) for v_0r in eig.network.v_0r)
            if shortest_r_dist > state.t
                # println("  EXCLUDING vi = $vi: shortest_r_dist = $shortest_r_dist")
                continue
            end
            
            push!(new_vs_just_before_cover[att_idx], vi)
        end
    end

    @constraint(LP_DOn, con4[y=1:length(att_supp)], z[y] ≤ sum( x[(p[1], p[2]), (p[1], p[2] + eig.δ)] for p in new_vs_just_before_cover[y]) )
    
    # @constraint(LP_DOn, con4[y=1:length(att_supp)], z[y] ≤ sum( ind_xvi_full[y, v, i] * x[(v, i - eig.δ), (v, i)]
    #         for v in nodes_1, i in Int(minimum(eig.dist_mtx[v_0r, v] / eig.speed_D for v_0r in eig.network.v_0r ) + eig.δ):Int(ceil(eig.t_max))))
    # # + sum( ind_xvi[x, v, 1] * y[(v, 0), (v, 1)] for v in unique_rs) )  # special i_idx = 1 for resource start states - UNNECESSARY, assuming v_0 ∉ v_0r
    
    @objective(LP_DOn, Max, sum( att_supp_probs[y] * z[y] for y=1:length(att_supp) ))  # prob of interdiction
    
    set_time_limit_sec(LP_DOn, timeout)

    # Fix !!FORCE y[(9, 90), (9, 91)] = 1
    # ???????????? ind_xvi_full[1, 9, 91.0] is 1
    # ???????????? ind_xvi_full[1, 17, 181.0] is 1
    # fix(y[(9, 90), (9, 91)], 1; force=true)
    
    optimize!(LP_DOn)
    
    if termination_status(LP_DOn) == TIME_LIMIT
        if primal_status(LP_DOn) != FEASIBLE_POINT
            println("Network DO timedout after ", timeout, " secs, but didn't find a feasible point")
            flush(stdout)
            # do something here? Like give it a bit more time to find a feasible point?
            return (new_def_strat=nothing,
                obj_value=nothing,
                LP_DO=LP_DOn,
                num_nodes_new_graph=num_nodes_new_graph,
                num_edges_new_graph=num_edges_new_graph,
                time_new_graph_construction=time_new_graph_construction)
            # return nothing, nothing, LP_DOn  # return essentially nothing
        elseif primal_status(LP_DOn) == FEASIBLE_POINT
            println("Network DO timedout after ", timeout, " secs, found a feasible point")
            flush(stdout)
        end
    else
        # no timeout
        if termination_status(LP_DOn) != OPTIMAL
            # compute conflict set
            compute_conflict!(LP_DOn)
            if get_attribute(LP_DOn, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
               iis_model, _ = copy_conflict(LP_DOn)
               print(iis_model)
            end
            error("Network DO did not solve to optimality and did not timeout. ", termination_status(LP_DOn))
        end
    end
    
#     for a=new_vs
#         if a == sink_node
#             println("CHECKING THE SINK")
#         end
        
#         # for b in arc_lists_o[a]
#         #     if value(x[a, b]) > 0
#         #         yeet = value(x[a, b])
#         #         println("Found x[$a, $b] = $yeet")
#         #     end
#         # end
        
#         for b in arc_lists_i[a]
#             if value(y[b, a]) > 0
#                 yeet = value(y[b, a])
#                 println("Found y[$b, $a] = $yeet")
#             end
#         end
        
#         if a == sink_node
#             println("FIN CHECKING THE SINK")
#         end
#     end
    
    # extract defender strategies
    # curr_flows = Dict((a, b) => value(y[a, b]) for a in new_graph.new_vs for b in new_graph.arc_lists_o[a])  # inefficient
    
    # Only deal with non-zero flows
    curr_flows = DefaultDict(0)
    for a in new_graph.new_vs
        for b in new_graph.arc_lists_o[a]
            temp = value(x[a, b])
            if temp > 0
                curr_flows[(a, b)] = temp
            end
        end
    end
    
    
    new_def_strat = [[] for _ in eig.network.v_0r]
    
    while sum(curr_flows[(source_node, b)] for b in arc_lists_o[source_node]) > 0.5
    # while sum(value(y[source_node, b]) for b in arc_lists_o[source_node]) > 0.5
        
        # find a source-sink path of non-zero flow, and send 1 virtual resource along it
        # min_arc_flow = length(eig.network.v_0r)  # minimum flow encountered along path
        seq_of_nodes = []  # sequence of nodes in time-ext. graph that flows visit

        curr_node = source_node  # current node in time-ext. graph

        # arcs_to_sub = []  # list of arcs whose flows will be subtracted
        
        counter_ = 0
        # while curr_node != sink_node  # !! WARNING, will not cover sink node, but no need since the sink node here does not correspond with a node in the og graph
        while curr_node != sink_node && counter_ < 1000  # !! WARNING, will not cover sink node, but no need since the sink node here does not correspond with a node in the og graph
            # find flow from (curr_node, curr_time, 1)
            for node in arc_lists_o[curr_node]
                if curr_flows[(curr_node, node)] > 0.5  # greedily select first outgoing arc with positive flow
                    # node is the next node in flow
                    push!(seq_of_nodes, node)
                    
#                     # update min_arc_flow
#                     if min_arc_flow > curr_flows[(curr_node, node)]
#                         min_arc_flow = curr_flows[(curr_node, node)]
#                     end
                    
                    # my_debug = curr_flows[(curr_node, node)]
                    # my_value = value(y[curr_node, node])
                    
                    # println("Found: old node is $curr_node new node is $node")
                    
                    # push!(arcs_to_sub, (curr_node, node))
                    
                    curr_flows[(curr_node, node)] -= 1
                    curr_node = node
                    
                    break;  # next in inner while-loop
                end
            end
            
            # display(curr_node)
            
            # if curr_node[1] == v_inf
            #     # path is done
            #     break;
            # end
            
            counter_ += 1
        end
        
        # println("DONE COUNTER, counter_ = $counter_")
        
#         # add sink node (since it is not covered in the above loop)
#         sink_arrival_time = -1.0
#         push!(new_att_strat, (v=v_inf, t=node[2]))
        
#         for b in arc_lists_i[sink_node]
#             println("Checking $b")
#             if value(x[b, sink_node]) > 0.5
#                 println("!!!! SINK: Found x[$b, $sink_node] = 1")
#                 sink_arrival_time = b[2]
#             end
#         end
#         push!(new_att_strat, (v=v_inf, t=sink_arrival_time))
        
        # outgoing_flow = sum(curr_flows[(source_node, b)] for b in arc_lists_o[source_node])
        
        # println("!!!!Done a path, sum of flows out of source is now: $outgoing_flow")   
        # println("seq_of_nodes is:")
        # display(seq_of_nodes)
        
        # extract resource schedule from seq_of_nodes. e.g. seq_of_nodes = [(4, 0), (7, 5), (2, 6), (2, 9), (3, 18), SINK]
        res_schd = []  # resource schedule (so int times); start at some (v_0r, 0, t^+)  (use float for type consistency)
        curr_vertex = seq_of_nodes[1][1]  # first entry is node
        # depart_time = 0  # departure time of the current state (to be computed)
        arrival_time = 0  # arrival time of the current state (to be computed)
        
        for node_s in seq_of_nodes[1:end-1]  # node_s is of the form (v, t); exclude sink_node
            if node_s[1] != curr_vertex
                # have bias towards waiting at the curr_vertex for longer, if node_s[2] is greater than depart_time + d(curr_vertex, node_s[1]) / s_D
                biased_depart_time = node_s[2] - eig.dist_mtx[curr_vertex, node_s[1]] / eig.speed_D  # wait as long as possible, then go max speed to reach the next node in time
                
                push!(res_schd, (v=curr_vertex, t_a=arrival_time, t_b=biased_depart_time))
                
                # update vertex and arrival time
                curr_vertex = node_s[1]
                arrival_time = node_s[2]
            end
        end
        
        # add last state
        push!(res_schd, (v=curr_vertex, t_a=arrival_time, t_b=Int(ceil(eig.t_max))))
        
        # println("ADDING SCHEDULE:")
        # display(res_schd)
        
        # push!(new_def_strat, res_schd)  # place resource schedule into defender strategy

        # ensure new_def_strat[r] starts at eig.network.v_0r[r]  (maybe this is assumed somewhere in the code, can't remember)
        flag_successful = false
        for (idx, res_node) in enumerate(eig.network.v_0r)
            if isempty(new_def_strat[idx]) && res_schd[1].v == res_node
                new_def_strat[idx] = res_schd
                flag_successful = true
                break
            end
        end

        if !flag_successful
            error("Oh no, we couldn't insert the new resource schedule into new_def_strat. new_def_strat: $new_def_strat and res_schd = $res_schd")
        end
        
    end  # endwhile
    
    obj_value = objective_value(LP_DOn)
    
    # return def_strat, obj_value, LP_DOn

    return (new_def_strat=new_def_strat,
        obj_value=obj_value,
        LP_DO=LP_DOn,
        num_nodes_new_graph=num_nodes_new_graph,
        num_edges_new_graph=num_edges_new_graph,
        time_new_graph_construction=time_new_graph_construction)
end
# -

# ### Data Loading

function converttrial2filepath(size::Int64, trial::Int64)
    """
    size is any integer in [3, 10].
    trial values:
        [1, 10] if size in {3, 5, 6, 7, 9}
        [1, 11] if size in {4, 8, 10}
    """
    @assert(1 <= trial <= 11 && 3 <= size <= 10, "Invalid size and trial pair")
    @assert((trial == 11) <= (size in [4, 8, 10]), "Invalid size and trial pair (case 2)")
    
    return "./Dataset_VNS_CAIE_Paper/" * string(size) * "/T" * string(trial) * "/Game_Parameters/"
end

function getEIGfromfile(filepath; use_orig_t_max=true, t_max_offset=0.75, force_speed_A=nothing, force_speed_D=nothing)
    """
    Returns an EIG from loading data from a filepath (filepath leads to a "Game_Parameters" folder)

    t_max_offset: See `use_orig_t_max` (does nothing if use_orig_t_max==true)
    use_orig_t_max: if true, uses the original t_max in the data file; else will compute t_max according to:
        t_max = dist_mtx[v_0, v_∞] / speed_A + t_max_offset
    force_speed_A: if not nothing, will manually force the speed_A EIG parameter to force_speed_A
    force_speed_D: if not nothing, will manually force the speed_D EIG parameter to force_speed_D
    """
    @assert(isinteger(t_max_offset - 0.75) && t_max_offset - 0.75 ≥ 0, "t_max_offset must be of the form `n + 0.75` for non-negative integer n")
    
    data_num_nodes = CSV.File(filepath * "node.csv", header=["col1"])
    data_points = CSV.File(filepath * "points.csv", header=["node1", "node2", "dist"])
    data_v_0 = CSV.File(filepath * "Crime.csv", header=["col1"])
    data_v_0r = CSV.File(filepath * "police_station.csv", header=["col1"])
    data_exit_nodes = CSV.File(filepath * "Virtual_Exit.csv", header=["col1"])
    data_v_∞ = CSV.File(filepath * "exit.csv", header=["col1"])
    data_t_max = CSV.File(filepath * "Time_Limits.csv", header=["col1"])

    # weird formatting with speeds; so do it this way
    if isnothing(force_speed_D)
        data_speed_D = parse(Int64, CSV.File(filepath * "Player_Speeds.csv")[1].var"Defender Speed")  # defender speed
    else
        data_speed_D = force_speed_D
    end
    if isnothing(force_speed_A)
        data_speed_A = CSV.File(filepath * "Player_Speeds.csv", skipto=4)[1].var"Defender Speed"  # attacker speed (weird formatting of orig file)
    else
        data_speed_A = force_speed_A
    end

    speed_lcm = lcm(data_speed_A, data_speed_D)  # scale time by lcm(s_A, s_D) (so that all travel times are integral)
    
    num_nodes = data_num_nodes[1].col1
    edges = [Edge(row[1], row[2], row[3] * speed_lcm) for row in data_points]  # SPEED ADJUSTING: scale edges by lcm(s_A, s_D)
    v_0 = data_v_0[1].col1  # single int
    v_0r = [x.col1 for x in data_v_0r] # vector of ints
    exit_nodes = [x.col1 for x in data_exit_nodes] # vector of ints
    v_∞ = data_v_∞[1].col1  # single int
    T_MAX = data_t_max[1].col1 * speed_lcm  # single int. SPEED ADJUSTING: scale by lcm(s_A, s_D)

    @assert(num_nodes == v_∞, "num_nodes should be equal to v_inf: filepath=$filepath")
    
    # DONT adjust speeds ()
    speed_D = data_speed_D
    speed_A = data_speed_A
    
    data_eig_networkgraph = NetworkGraph(num_nodes, edges, v_0, v_0r, exit_nodes, v_∞)
    
    dist_mtx = computedistancematrix(data_eig_networkgraph);  # note this is distances scaled by lcm(s_A*s_D) of the original edge length - must divide by suitable speeds to get travel times
    neighbourhoods = computeneighbourhoods(data_eig_networkgraph);
    
    DELTA = 1  # in the scaled version, this is the 'optimal' delta (note the corresponding opt delta for the original EIG is 1/lcm(s_A, s_D) )
    # T_MAX = maximum(filter(!isinf, dist_mtx)) + 0.75

    if use_orig_t_max
        T_MAX = T_MAX - 0.25  # so that our half-int strategies and ϵ in big-M constraints work
        @assert(dist_mtx[v_0, v_∞] / speed_A <= T_MAX)
    else
        # compute according to t_max_offset (quickest time for attacker to get to v_∞, then add t_max_offset)
        T_MAX = dist_mtx[v_0, v_∞] / speed_A + t_max_offset
    end

    # check if feasible attacker strategy exists
    if T_MAX < div(dist_mtx[v_0, v_∞], speed_A)
        println("!! Making t_max = dist_mtx[v_0, v_∞] / speed_A + $t_max_offset")
        println("!! EIG is attacker-infeasible !! Making t_max = dist_mtx[v_0, v_∞] / s_A + 0.75")
        flush(stdout)
        T_MAX = dist_mtx[v_0, v_∞] / speed_A + 0.75
    end
    
    the_eig = EIG(data_eig_networkgraph, DELTA, T_MAX, dist_mtx, neighbourhoods, speed_A, speed_D)
    
    return the_eig
end

# ### Do run of a trial

function do_run_of_trial_custom(grid_size, trial_num; A_oracle_num=2, D_oracle_num=2, total_timeout=300, A_timeout=30, D_timeout=30, printing=0, silent_solvers=true, t_max_offset=0.75, use_orig_t_max=true, abstol=1e-6)
    """
    Do a run on a specified example given by grid_size and trial_num with specified oracles.
    
    grid_size: Int in [3, 10]
    trial_num: Int in [1, 10]
    A_oracle_num: Int in {1, 2}. Determines attacker oracle
        If 1 then Zhang attacker oracle. If 2 then network attacker oracle.
    D_oracle_num: Int in {1, 2}. Determines defender oracle
        If 1 then Zhang defender oracle. If 2 then network defender oracle.
    """
    @assert(3 <= grid_size <= 10, "Invalid grid_size")
    @assert(1 <= trial_num <= 10, "Invalid trial_num")
    @assert(A_oracle_num in [1, 2], "A_oracle_num must be in [1, 2]")
    @assert(D_oracle_num in [1, 2], "D_oracle_num must be in [1, 2]")
    
    A_ORACLE_ID_TO_ORACLE = [attackeroraclezhang, attackeroraclenew]
    D_ORACLE_ID_TO_ORACLE = [defenderoraclezhang, defenderoraclenew]
    
    filepath = converttrial2filepath(grid_size, trial_num)
    the_eig = getEIGfromfile(filepath, t_max_offset=t_max_offset, use_orig_t_max=use_orig_t_max)
    
    # the_eig.t_max = the_eig.dist_mtx[the_eig.network.v_0, the_eig.network.v_∞] + 0.75
    
    # compute simple heuristic strategies (shortest path for attacker, stay at stations for defender)
    att_heu_strat = makeheuristicattacker(the_eig)
    def_heu_strat = makedefstrat_endnodes(the_eig)

    # network oracles; reset att_strats and def_strats
    att_strats = [att_strat_to_A_half_strat(the_eig.dist_mtx, att_heu_strat, the_eig.speed_A)]
    def_strats = [def_heu_strat]

    if A_oracle_num == 2 && D_oracle_num == 2
        println("!!!!!! STARTING grid_size=$grid_size and trial_num=$trial_num NETWORK DOUBLE ORACLE")
    elseif A_oracle_num == 1 && D_oracle_num == 1
        println("!!!!!! STARTING grid_size=$grid_size and trial_num=$trial_num ZHANG DOUBLE ORACLE")
    else
        println("!!!!!! STARTING grid_size=$grid_size and trial_num=$trial_num CUSTOM DOUBLE ORACLE")
    end
    flush(stdout)
    
    results = @timed EIGSzhang!(the_eig, att_strats, def_strats; DO=D_ORACLE_ID_TO_ORACLE[D_oracle_num], AO=A_ORACLE_ID_TO_ORACLE[A_oracle_num],
        total_timeout=total_timeout, A_timeout=A_timeout, D_timeout=D_timeout, printing=printing, silent_solvers=silent_solvers, abstol=abstol)
    time = results.time
    results = results.value

    t_max_info = (
        use_orig_t_max=use_orig_t_max,
        t_max_scaled_and_offset=the_eig.t_max,
        t_max_orig=CSV.File(filepath * "Time_Limits.csv", header=["col1"])[1].col1
        # t_max_offset = use_orig_t_max ? -0.25 : t_max_offset
    )

    speed_info = (
        speed_A=the_eig.speed_A,
        speed_D=the_eig.speed_D
    )
    
    # return data
    return (
        time=time,
        results=results,
        AO=string(A_ORACLE_ID_TO_ORACLE[A_oracle_num]),
        DO=string(D_ORACLE_ID_TO_ORACLE[D_oracle_num]),
        total_timeout=total_timeout,
        abstol=abstol,
        t_max_info=t_max_info,
        speed_info=speed_info
    )
    
        # att_strats
        # att_probs
        # def_strats
        # def_probs
        # obj_core_over_time
        # obj_DO_over_time
        # obj_AO_over_time
        # coreLP_time_over_time
        # DO_time_over_time
        # AO_time_over_time
        # convergence_flag
        # construction times
end

# +
# function do_run_of_trial(grid_size, trial_num; total_timeout_zh=300, total_timeout_ne=300, A_timeout_zh=30, A_timeout_ne=30, D_timeout_zh=30, D_timeout_ne=30, printing=0, silent_solvers=true, t_max_offset=0.75, use_orig_t_max=true)
#     """
#     Do a run (both Zhang and network oracles) on a specified example given by grid_size and trial_num
    
#     grid_size: Int in [3, 10]
#     trial_num: Int in [1, 10]
#     """
#     @assert(3 <= grid_size <= 10, "Invalid grid_size")
#     @assert(1 <= trial_num <= 10, "Invalid trial_num")

#     results_and_time_zh = do_run_of_trial_custom(grid_size, trial_num, A_oracle_num=1, D_oracle_num=1,
#         total_timeout=total_timeout_zh, A_timeout=A_timeout_zh, D_timeout=D_timeout_zh,
#         printing=printing, silent_solvers=silent_solvers, t_max_offset=t_max_offset, use_orig_t_max=use_orig_t_max)
    
#     results_and_time_ne = do_run_of_trial_custom(grid_size, trial_num, A_oracle_num=2, D_oracle_num=2,
#         total_timeout=total_timeout_ne, A_timeout=A_timeout_ne, D_timeout=D_timeout_ne,
#         printing=printing, silent_solvers=silent_solvers, t_max_offset=t_max_offset, use_orig_t_max=use_orig_t_max)

#     return (
#         time_zh=results_and_time_zh.time,
#         results_zh=results_and_time_zh.results,
#         time_ne=results_and_time_ne.time,
#         results_ne=results_and_time_ne.results
#     )
    
#     # println("\n!!!!!!!! STARTING grid_size=$grid_size and trial_num=$trial_num")
#     # flush(stdout)
    
#     # filepath = converttrial2filepath(grid_size, trial_num)
#     # the_eig = getEIGfromfile(filepath, t_max_offset=t_max_offset, use_orig_t_max=use_orig_t_max)
    
#     # # the_eig.t_max = the_eig.dist_mtx[the_eig.network.v_0, the_eig.network.v_∞] + 0.75
    
#     # # compute simple heuristic strategies (shortest path for attacker, stay at stations for defender)
#     # att_heu_strat = makeheuristicattacker(the_eig)
#     # def_heu_strat = makedefstrat_endnodes(the_eig)

#     # # network oracles; reset att_strats and def_strats
#     # att_strats = [att_strat_to_A_half_strat(the_eig.dist_mtx, att_heu_strat, the_eig.speed_A)]
#     # def_strats = [def_heu_strat]
    
#     # println("!!!!!! STARTING NETWORK DOUBLE ORACLE")
#     # flush(stdout)
    
#     # results_ne = @timed EIGSzhang!(the_eig, att_strats, def_strats; DO=defenderoraclenew, AO=attackeroraclenew,
#     #     total_timeout=total_timeout_ne, A_timeout=A_timeout_ne, D_timeout=D_timeout_ne, printing=printing, silent_solvers=silent_solvers)
#     # time_ne = results_ne.time
#     # results_ne = results_ne.value

#     # # Zhang oracles
#     # att_strats = [att_heu_strat]
#     # def_strats = [def_heu_strat]
    
#     # println("!!!!!! STARTING ZHANG DOUBLE ORACLE")
#     # flush(stdout)
    
#     # results_zh = @timed EIGSzhang!(the_eig, att_strats, def_strats; DO=defenderoraclezhang, AO=attackeroraclezhang,
#     #     total_timeout=total_timeout_zh, A_timeout=A_timeout_zh, D_timeout=D_timeout_zh, printing=printing, silent_solvers=silent_solvers)
#     # time_zh = results_zh.time
#     # results_zh = results_zh.value
    
#     # # return data
#     # return (
#     #     time_zh=time_zh,
#     #     results_zh=results_zh,
#     #     time_ne=time_ne,
#     #     results_ne=results_ne
#     # )
    
#         # att_strats
#         # att_probs
#         # def_strats
#         # def_probs
#         # obj_core_over_time
#         # obj_DO_over_time
#         # obj_AO_over_time
#         # coreLP_time_over_time
#         # DO_time_over_time
#         # AO_time_over_time
#         # convergence_flag
#         # construction times
# end

# +
# function do_run_of_trial_and_write_result(grid_size, trial_num; total_timeout_zh=300, total_timeout_ne=300, A_timeout_zh=30, A_timeout_ne=30, D_timeout_zh=30, D_timeout_ne=30, printing=0, silent_solvers=true, t_max_offset=0.75, use_orig_t_max=true)
#     results = do_run_of_trial(grid_size, trial_num;
#         total_timeout_zh=total_timeout_zh, total_timeout_ne=total_timeout_ne,
#         A_timeout_zh=A_timeout_zh, A_timeout_ne=A_timeout_ne,
#         D_timeout_zh=D_timeout_zh, D_timeout_ne=D_timeout_ne, printing=printing, silent_solvers=silent_solvers, t_max_offset=t_max_offset, use_orig_t_max=use_orig_t_max)
    
#     # write results to file
#     if use_orig_t_max
#         file_string = "run_results/size" * string(grid_size) * "_trial" * string(trial_num) * "_origtmax.json"
#     else
#         file_string = "run_results/size" * string(grid_size) * "_trial" * string(trial_num) * "_tmaxoff" * string(Int(t_max_offset * 100)) * ".json"
#     end
#     open(file_string, "w") do f
#         JSON.print(f, results)
#     end
# end
# -

function do_run_of_trial_and_write_result_custom(grid_size, trial_num; A_oracle_num=2, D_oracle_num=2, total_timeout=300, A_timeout=30, D_timeout=30, printing=0, silent_solvers=true, t_max_offset=0.75, use_orig_t_max=true, dirname="run_results", abstol=1e-6)
    @assert(A_oracle_num in [1, 2], "Attacker oracle num must be in [1, 2] (1 for Zhang, 2 for network)")
    @assert(D_oracle_num in [1, 2], "Defender oracle num must be in [1, 2] (1 for Zhang, 2 for network)")
    @assert(A_oracle_num == D_oracle_num, "Should be using same type oracle in final runs")
    
    results = do_run_of_trial_custom(grid_size, trial_num, A_oracle_num=A_oracle_num, D_oracle_num=D_oracle_num,
        total_timeout=total_timeout, A_timeout=A_timeout, D_timeout=D_timeout,
        printing=printing, silent_solvers=silent_solvers, t_max_offset=t_max_offset, use_orig_t_max=use_orig_t_max, abstol=abstol)
    
    A_ORACLE_ID_TO_ORACLE = [attackeroraclezhang, attackeroraclenew]
    D_ORACLE_ID_TO_ORACLE = [defenderoraclezhang, defenderoraclenew]

    oracle_str = A_oracle_num == 1 ? "zhang" : "network"
    
    # write results to file
    if use_orig_t_max
        # file_string = "run_results_JUST_NETWORK/size" * string(grid_size) * "_trial" * string(trial_num) * "_origtmax.json"
        file_string = dirname * "_" * oracle_str * "/size" * string(grid_size) * "_trial" * string(trial_num) * "_origtmax.json"
    else
        # file_string = "run_results_JUST_NETWORK/size" * string(grid_size) * "_trial" * string(trial_num) * "_tmaxoff" * string(Int(t_max_offset * 100)) * ".json"
        file_string = dirname * "_" * oracle_str * "/size" * string(grid_size) * "_trial" * string(trial_num) * "_tmaxoff" * string(Int(t_max_offset * 100)) * ".json"
    end

    if !isdir(dirname * "_" * oracle_str)
        mkdir(dirname * "_" * oracle_str)
    end
    open(file_string, "w") do f
        JSON.print(f, results)
    end
end

# T_MAX_OFFSETS_CS_LIST = [75, 875, 1075, 1675, 2075, 3075]  # list of t_max offsets (centiseconds) used: OLD, now using seq(0 10 60)

dirname = "run_results_final4"  # hardcoded

# to get compilation
do_run_of_trial_and_write_result_custom(10, 8, A_oracle_num=2, D_oracle_num=2,
        total_timeout=3600, A_timeout=nothing, D_timeout=nothing,
        printing=false, silent_solvers=true,
        t_max_offset=0 + 0.75, use_orig_t_max=true, dirname=dirname, abstol=1e-7)

for GRID_SIZE in 3:10
    for TRIAL_NUM in 1:10
        do_run_of_trial_and_write_result_custom(GRID_SIZE, TRIAL_NUM, A_oracle_num=2, D_oracle_num=2,
                total_timeout=3600, A_timeout=nothing, D_timeout=nothing,
                printing=false, silent_solvers=true,
                t_max_offset=0 + 0.75, use_orig_t_max=true, dirname=dirname, abstol=1e-7)
    end
end

# do_run_of_trial_and_write_result_custom(7, 7, A_oracle_num=2, D_oracle_num=2,
#         total_timeout=3600*24*4, A_timeout=nothing, D_timeout=nothing,
#         printing=false, silent_solvers=true,
#         t_max_offset=0 + 0.75, use_orig_t_max=true, dirname="run_results_long_custom", abstol=1e-5)

# do_run_of_trial_and_write_result_custom(3, 1, A_oracle_num=2, D_oracle_num=2,
#         total_timeout=3600, A_timeout=nothing, D_timeout=nothing,
#         printing=false, silent_solvers=true,
#         t_max_offset=0 + 0.75, use_orig_t_max=true, dirname=dirname, abstol=1e-7)

# # ## ------------------------------ MAIN ------------------------------ (run individual grid-trial pair)
# # usage: EIG_Notebook.jl <total_timeout for each oracle int> <t_max_offset int> <use_orig_t_max bool> <grid_size> <trial_num> <A_oracle_num> <D_oracle_num>
# # if A_oracle_num = 1 then Zhang attacker oracle is used, if A_oracle_num = 2 then network attacker oracle is used
# # if D_oracle_num = 1 then Zhang defender oracle is used, if D_oracle_num = 2 then network defender oracle is used

# ULT_TIMEOUT = parse(Int, ARGS[1])
# T_MAX_OFFSET_INT = parse(Int, ARGS[2])
# USE_ORIG_T_MAX = parse(Bool, ARGS[3])
# GRID_SIZE = parse(Int, ARGS[4])
# TRIAL_NUM = parse(Int, ARGS[5])
# A_ORACLE_NUM = parse(Int, ARGS[6])
# D_ORACLE_NUM = parse(Int, ARGS[7])


# do_run_of_trial_and_write_result_custom(GRID_SIZE, TRIAL_NUM, A_oracle_num=A_ORACLE_NUM, D_oracle_num=D_ORACLE_NUM,
#                 total_timeout=ULT_TIMEOUT, A_timeout=nothing, D_timeout=nothing,
#                 printing=false, silent_solvers=true,
#                 t_max_offset=T_MAX_OFFSET_INT + 0.75, use_orig_t_max=USE_ORIG_T_MAX, dirname=dirname, abstol=1e-7)
