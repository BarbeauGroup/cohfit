import numpy as np
from numba import njit

@njit
def rebin_histogram(counts, bin_edges, new_bin_edges):
    new_counts = np.zeros(len(new_bin_edges) - 1, dtype=np.float64)

    for i in range(len(new_bin_edges) - 1):
        new_bin_start, new_bin_end = new_bin_edges[i], new_bin_edges[i+1]
        
        # Loop over each fine bin
        for j in range(len(bin_edges) - 1):
            fine_bin_start, fine_bin_end = bin_edges[j], bin_edges[j+1]

            if fine_bin_end <= new_bin_start:
                continue
            if fine_bin_start >= new_bin_end:
                break
            
            # Check for overlap between fine bin and new bin
            overlap_start = max(new_bin_start, fine_bin_start)
            overlap_end = min(new_bin_end, fine_bin_end)
            
            if overlap_start < overlap_end:
                # Calculate the overlap width
                overlap_width = overlap_end - overlap_start
                fine_bin_width = fine_bin_end - fine_bin_start
                
                # Proportion of the fine bin's count that goes into the new bin
                contribution = (overlap_width / fine_bin_width) * counts[j]
                
                # Add the contribution to the new bin's count
                new_counts[i] += contribution
    
    return new_counts

@njit
def rebin_histogram2d(counts, x_bin_edges, y_bin_edges, new_x_bin_edges, new_y_bin_edges):
    new_counts = np.zeros((len(new_x_bin_edges) - 1, len(new_y_bin_edges) - 1), dtype=np.float64)

    for i in range(len(new_x_bin_edges) - 1):
        new_x_bin_start, new_x_bin_end = new_x_bin_edges[i], new_x_bin_edges[i+1]
        
        for j in range(len(new_y_bin_edges) - 1):
            new_y_bin_start, new_y_bin_end = new_y_bin_edges[j], new_y_bin_edges[j+1]

            for x in range(len(x_bin_edges) - 1):
                x_bin_start, x_bin_end = x_bin_edges[x], x_bin_edges[x+1]

                if x_bin_end <= new_x_bin_start:
                    continue
                if x_bin_start >= new_x_bin_end:
                    break

                for y in range(len(y_bin_edges) - 1):
                    y_bin_start, y_bin_end = y_bin_edges[y], y_bin_edges[y+1]

                    if y_bin_end <= new_y_bin_start:
                        continue
                    if y_bin_start >= new_y_bin_end:
                        break

                    overlap_x_start = max(new_x_bin_start, x_bin_start)
                    overlap_x_end = min(new_x_bin_end, x_bin_end)
                    overlap_y_start = max(new_y_bin_start, y_bin_start)
                    overlap_y_end = min(new_y_bin_end, y_bin_end)

                    if overlap_x_start < overlap_x_end and overlap_y_start < overlap_y_end:
                        overlap_x_width = overlap_x_end - overlap_x_start
                        x_bin_width = x_bin_end - x_bin_start
                        overlap_y_width = overlap_y_end - overlap_y_start
                        y_bin_width = y_bin_end - y_bin_start

                        contribution = (overlap_x_width / x_bin_width) * (overlap_y_width / y_bin_width) * counts[x, y]

                        new_counts[i, j] += contribution
    
    return new_counts

def centers_to_edges(centers):
    return np.concatenate((centers - (centers[1] - centers[0]) / 2, [centers[-1] + (centers[1] - centers[0]) / 2]))

def edges_to_centers(edges):
    edges = np.asarray(edges)
    return (edges[1:] + edges[:-1]) / 2
    """
    Calculate the total counts in an n-dimensional histogram with uneven bin widths.

    Parameters:
        histogram (ndarray): An n-dimensional array of histogram counts.
        bins (list of arrays): A list of arrays, each representing the bin edges along one dimension.

    Returns:
        total_counts (float): The total counts in the histogram.
    """
    # Validate dimensions
    if len(bins) != histogram.ndim:
        raise ValueError("Number of bin arrays must match the dimensionality of the histogram.")

    # Calculate bin widths along each dimension
    bin_widths = [np.diff(b) for b in bins]

    # Create a meshgrid of bin widths
    bin_width_mesh = np.meshgrid(*bin_widths, indexing='ij')

    # Calculate bin volumes (product of widths along each dimension)
    bin_volumes = histogram
    for i in range(len(bin_width_mesh)):
        bin_volumes *= bin_width_mesh[i]

    # Multiply histogram counts by bin volumes to get total counts
    total_counts = np.sum(bin_volumes)

    return total_counts