import torch

def check_collisions(points):
    """
    points: List of points (x, y, radius) represented as PyTorch tensors
    """
    collisions = []
    collision_flags = []
    collision_flag_any = False
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            circle1 = points[i]
            circle2 = points[j]

            # Calculate the distance between the centers of the circles
            center1 = circle1[:2]
            center2 = circle2[:2]
            distance = torch.norm(center1 - center2)

            # Check if the distance is less than or equal to the sum of the radii
            if distance <= circle1[2] + circle2[2]:
                collision_flag_any = True
                return collision_flag_any
                collisions.append((circle1, circle2))
                collision_flags.append(1)
            # else:
            #     collision_flags.append(0)
    # return collision_flags, collision_flag_any
    return collision_flag_any

# Example usage:
if __name__ == "__main__":
    # List of points (x, y, radius) represented as PyTorch tensors
    points = [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([3.0, 3.0, 1.0]), torch.tensor([5.0, 5.0, 1.0]), torch.tensor([2.0, 2.0, 1.0])]
