import os
import csv
import matplotlib.pyplot as plt
import mplcursors
import ast

def canonicalize_change(key_str, dim=3):
    left, cycles = ast.literal_eval(key_str)
    left = tuple(left)
    cycles = tuple(cycles)

    if dim == 2:
        assert len(left) >= 4, "Expected at least 4 elements for 2D change"
        left = left[:4]  # keep only edge and triangle info

    # Swap every pair in the left tuple
    left_pairs = [(left[i], left[i+1]) for i in range(0, len(left), 2)]
    dual_left_pairs = [(b, a) for (a, b) in left_pairs]
    dual_left = tuple(x for pair in dual_left_pairs for x in pair)

    # Swap cycles
    dual_cycles = (cycles[1], cycles[0])

    canonical = min((left, cycles), (dual_left, dual_cycles))
    return str(canonical)


def load_data_from_folder(
        folder_path,
        num_sensors,
        total_counts,
        canonicalize=True,
        clean=0.0,
        dim=3,
        r_max=None):
    radii = []
    data_by_key = {}

    csv_files = []
    for f in os.listdir(folder_path):
        if f.endswith(f'num_{num_sensors}.csv'):
            try:
                # Extract radius from filename like 'radius_0.45_num_30.csv'
                radius_part = f.split("_")[1]  # '0.45'
                radius_val = float(radius_part)
                if r_max is None or radius_val < r_max:
                    csv_files.append((radius_val, f))
            except Exception as e:
                print(f"Skipping file {f}: {e}")

    csv_files.sort()
    csv_files = [f for (_, f) in csv_files]

    for fname in csv_files:
        filepath = os.path.join(folder_path, fname)
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

            # rows[0] should be something like ["Radius:", "0.45"]
            # rows[1] might be ["Number:", "30"] (we don't strictly need that)
            # rows[2] should be ["Element", "Count"]

            first_line = rows[0]
            # Typically, first_line[1] should be the numeric radius
            radius_value = float(first_line[1])
            radii.append(radius_value)

            for row in rows[3:]:
                raw_key_str = row[0]
                count = int(row[1])
                if fname == "radius_0.45_num_30.csv":
                    print(raw_key_str)
                    print(count)
                prob = count / total_counts

                if canonicalize:
                    key_str = canonicalize_change(raw_key_str, dim=dim)
                else:
                    key_str = raw_key_str

                if key_str not in data_by_key:
                    data_by_key[key_str] = {}
                    data_by_key[key_str][radius_value] = prob

                if radius_value in data_by_key[key_str]:
                    data_by_key[key_str][radius_value] += prob  # aggregate if already exists
                else:
                    data_by_key[key_str][radius_value] = prob

    radii = sorted(radii)

    final_data_by_key = {}
    for key_str, radius_dict in data_by_key.items():
        counts_for_key = [radius_dict.get(r, 0) for r in radii]

        # Keep this key only if it has at least one value ≥ clean threshold
        if any(p >= clean for p in counts_for_key):
            final_data_by_key[key_str] = counts_for_key

    return radii, final_data_by_key

def plot_element_counts(radii, data_by_key, analyze=True, named_acs=True, dim=3):
    plt.figure(figsize=(10, 6))

    for key_str, counts in data_by_key.items():
        if named_acs:
            label = describe_atomic_change(key_str, dim=dim)
        else:
            label=key_str
        plt.plot(radii, counts, label=label, marker='o')

    plt.xlabel("Radius")
    plt.ylabel("Probability")
    plt.title(f"{dim}D Atomic Changes Observed by Radius")
    plt.tight_layout()

    if analyze:
        # Interactive cursor hover annotations
        import mplcursors
        crs = mplcursors.cursor(hover=True)

        @crs.connect("add")
        def _(sel):
            line = sel.artist
            x, y = sel.target
            key_str = line.get_label()
            sel.annotation.set_text(f"{key_str}\n({x:.3f}, {y:.3f})")
    else:
        plt.legend(loc='upper right', fontsize='small')

    plt.show()

def describe_atomic_change(key_str, dim=3):
    import ast
    left, cycles = ast.literal_eval(key_str)

    if dim == 2:
        edge_a, edge_r = left[0], left[1]
        tri_a, tri_r = left[2], left[3]
        tet_a, tet_r = 0, 0  # not present
    else:
        edge_a, edge_r = left[0], left[1]
        tri_a, tri_r = left[2], left[3]
        tet_a, tet_r = left[4], left[5]

    cyc_a, cyc_r = cycles

    parts = []
    if edge_a > 0 or edge_r > 0:
        parts.append(f"Edges: +{edge_a}, -{edge_r}")
    if tri_a > 0 or tri_r > 0:
        parts.append(f"Triangles: +{tri_a}, -{tri_r}")
    if dim == 3 and (tet_a > 0 or tet_r > 0):
        parts.append(f"Tetrahedra: +{tet_a}, -{tet_r}")
    if cyc_a > 0 or cyc_r > 0:
        parts.append(f"Cycles: +{cyc_a}, -{cyc_r}")

    return " | ".join(parts)


if __name__ == "__main__":
    folder = "output/2d_10000/"
    dim=2
    radii, data_by_key = load_data_from_folder(
        folder,
        num_sensors=30,
        total_counts=10000,
        canonicalize=True,
        clean=0.03,
        dim=dim,
        r_max=0.4)
    plot_element_counts(
        radii,
        data_by_key,
        analyze=False,
        dim=dim)

