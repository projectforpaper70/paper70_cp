# # Import necessary modules
# from sage.numerical.mip import MixedIntegerLinearProgram

# def duplicate_mip_problem(problem_old):
#     # Create a new mip_problem with another backend solver 
#     p = MixedIntegerLinearProgram(solver='GLPK' or "PPL" or "Gurobi")

#     # Get the constraints from the old problem
#     cons = problem_.constraints()

#     # Travel the cons list and extract the information needed
#     for constraint in cons:
#         # Extract variable indices and corresponding coefficients from cons_items
#         indices, coefficients = constraint[1]
        
#         # Use p.sum to create a linear equality constraint and add it to the new problem
#         p.add_constraint(p.sum(coefficients[i] * p[indices[i]] for i in range(len(indices))), min=constraint[0], max=constraint[2])
