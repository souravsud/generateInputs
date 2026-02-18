def generate_directions(sectors):
    """
    Generates evenly-spaced directions around 360 degrees.
    
    For CFD simulations, evenly distributed directions provide better coverage
    and reproducible results compared to random directions.
    
    Args:
        sectors (int): Number of direction sectors to generate.
                      For example, 16 sectors gives directions every 22.5°.
                      
    Returns:
        list: A list of evenly-spaced direction angles in degrees,
              starting from 0° (North) and going clockwise.
              
    Example:
        >>> generate_directions(4)
        [0, 90, 180, 270]
        >>> generate_directions(8)
        [0, 45, 90, 135, 180, 225, 270, 315]
    """
    sector_size = 360 // sectors
    return [i * sector_size for i in range(sectors)]
    
if __name__ == "__main__":
    # Example usage
    print("Evenly-spaced directions (4 sectors):", generate_directions(4))
    print("Evenly-spaced directions (16 sectors):", generate_directions(16))