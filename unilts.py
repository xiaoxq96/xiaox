# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:18:03 2022

@author: Xiaox
"""

def read_instance() -> dict:
    """Convert instance input files into raw problem data.

    Args:
        instance_path:  Path to the bin packing problem instance file.

    Returns:
        data: dictionary containing raw information for both bins and cases.

    """

    data = {"num_platte": 0, "platte_dim":[], "quantity": [], "stueck_ids": [],
            "stueck_lange": [], "stueck_breite":[]}

    with open("15x2x15x15.txt") as f:
        for i, line in enumerate(f):
            if i == 0:
                data["num_platte"] = int(line.split()[-1])
            elif i == 1:
                data["platte_dim"] = [int(i) for i in line.split()[-2:]]
            elif 2 <= i <= 4:
                continue
            else:
                case_info = list(map(int, line.split()))
                data["stueck_ids"].append(case_info[0])
                data["quantity"].append(case_info[1])
                data["stueck_lange"].append(case_info[2])
                data["stueck_breite"].append(case_info[3])

        return data
    
    
    def write_solution_to_file(solution_file_path: str,
                           cqm: dimod.ConstrainedQuadraticModel,
                           vars: "Variables",
                           sample: dimod.SampleSet,
                           cases: "Cases",
                           bins: "Bins",
                           effective_dimensions: list):
        """Write solution to a file.

        Args:
            solution_file_path: path to the output solution file. If doesn't exist,
                a new file is created.
            cqm: A ``dimod.CQM`` object that defines the 3D bin packing problem.
            vars: Instance of ``Variables`` that defines the complete set of variables
                for the 3D bin packing problem.
            sample: A ``dimod.SampleSet`` that represents the best feasible solution found.
            cases: Instance of ``Cases``, representing cases packed into containers.
            bins: Instance of ``Bins``, representing containers to pack cases into.
            effective_dimensions: List of case dimensions based on orientations of cases.

        """
        num_cases = cases.num_cases
        num_bins = bins.num_bins
        dx, dy, dz = effective_dimensions
        if num_bins > 1:
            num_bin_used = sum([vars.bin_on[j].energy(sample)
                                for j in range(num_bins)])
        else:
            num_bin_used = 1

        objective_value = cqm.objective.energy(sample)
        vs = [['case_id', 'bin-location', 'orientation', 'x', 'y', 'z', "x'",
               "y'", "z'"]]
        for i in range(num_cases):
            vs.append([cases.case_ids[i],
                       int(sum((j + 1) * vars.bin_loc[i, j].energy(sample)
                               if num_bins > 1 else 1
                               for j in range(num_bins))),
                       int(sum((r + 1) * vars.o[i, r].energy(sample) for r in
                               range(6))),
                       np.round(vars.x[i].energy(sample), 2),
                       np.round(vars.y[i].energy(sample), 2),
                       np.round(vars.z[i].energy(sample), 2),
                       np.round(dx[i].energy(sample), 2),
                       np.round(dy[i].energy(sample), 2),
                       np.round(dz[i].energy(sample), 2)])

        with open(solution_file_path, 'w') as f:
            f.write('# Number of bins used: ' + str(int(num_bin_used)) + '\n')
            f.write('# Number of cases packed: ' + str(int(num_cases)) + '\n')
            f.write(
                '# Objective value: ' + str(np.round(objective_value, 3)) + '\n\n')
            f.write(tabulate(vs, headers="firstrow"))
            f.close()
            print(f'Saved solution to '
                  f'{os.path.join(os.getcwd(), solution_file_path)}')