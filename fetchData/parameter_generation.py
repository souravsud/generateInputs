def generate_directions(sectors):
    """
    Generates evenly-spaced directions around 360 degrees.
    
    For CFD simulations, evenly distributed directions provide better coverage
    and reproducible results compared to random directions.
    
    Note: Uses floating-point division to ensure perfect 360° coverage with
    uniform spacing, even when 360 is not evenly divisible by the number of
    sectors (e.g., 7 sectors gives 51.43° spacing).
    
    Args:
        sectors (int): Number of direction sectors to generate.
                      For example, 16 sectors gives directions every 22.5°.
                      
    Returns:
        list: A list of evenly-spaced direction angles in degrees (floats),
              starting from 0° (North) and going clockwise.
              
    Example:
        >>> generate_directions(4)
        [0.0, 90.0, 180.0, 270.0]
        >>> generate_directions(8)
        [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    """
    # Using floating-point division ensures uniform spacing for all sector counts
    return [i * 360.0 / sectors for i in range(sectors)]
    
if __name__ == "__main__":
    # Example usage
    print("Evenly-spaced directions (4 sectors):", generate_directions(4))
    print("Evenly-spaced directions (16 sectors):", generate_directions(16))