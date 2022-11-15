"""
Tacker system from the Component 2
"""

import numpy as np


class Tracker:
    def __init__(self, objective_classes=None):
        """
        Args:
            objective_classes (list[int]): All id objective classes
        """
        objective_classes = [0] if objective_classes is None else objective_classes
        self.objective_classes = objective_classes

    def get_distance_object(self, main_position, object_positions, classes):
        """
        Calculate euclidean distance between main position and all objects

        Args:
            main_position (tuple[float, float]): Mains position to be compared with other objects (X, Y)
            object_positions (list[tuple[float, float]]): List of object positions (X, Y)
            classes (list[int]): Classes of each box detected
        Returns:
            tuple[int, list[float]]:
                closest_object: Index of the closest object to the main position
                all_distances: Distances between all object to the main position
        """
        only_objective = self._get_only_objective(object_positions, classes)
        only_inside_range = self._get_only_inside_range(only_objective)

        all_distances = []
        if len(only_inside_range) > 0:
            for _, position in only_inside_range:
                # Euclidean distance
                all_distances.append(np.linalg.norm(main_position - position))
            return only_inside_range[np.argmin(all_distances)][0], all_distances
        return 0, all_distances

    def _get_only_objective(self, object_positions, classes):
        """
        Get only positions that are objective

        Args:
            object_positions (list[tuple[float, float]]): List of object positions (X, Y)
            classes (list[int]): Classes of each box detected
        Returns:
            list[int, list[tuple[float, float]]]:
                idx: Index of object position in original list
                object_positions: List of object positions (X, Y)
        """
        only_objective = []
        for idx, value in enumerate(classes):
            if value in self.objective_classes:
                only_objective.append([idx, object_positions[idx]])
        return only_objective

    @staticmethod
    def _get_only_inside_range(only_objective):
        """
        Get only positions that are in center of the image

        Args:
            only_objective (list[int, list[tuple[float, float]]]): List idx in original list and object positions (X, Y)
        Returns:
            list[int, list[tuple[float, float]]]:
                idx: Index of object position in original list
                object_positions: List of object positions (X, Y)
        """
        only_inside_range = []
        for obj in only_objective:
            if 0.15 < obj[1][0] < 0.75 and 0.15 < obj[1][1] < 0.70:
                only_inside_range.append(obj)
        return only_inside_range
