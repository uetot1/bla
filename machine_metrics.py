import json
import math


def load_points(path):
    with open(path, encoding='utf-8') as file:
        data = json.load(file)
    return data['points']


def bd_rate_map(anchor_points, test_points, map_key='map50_95'):
    """Return average bitrate difference (%) at equal mAP; negative is better."""
    def curve(points):
        rates = [float(point['bitrate_kbps']) for point in points]
        maps = [float(point[map_key]) for point in points]
        if (len(points) != 4 or any(rate <= 0 for rate in rates) or
                len(set(rates)) != 4 or len(set(maps)) != 4 or
                any(not 0 <= value <= 1 for value in maps) or
                not all(math.isfinite(value) for value in rates + maps)):
            raise ValueError(
                'BD-rate-mAP requires four unique positive rates and four unique finite mAP values in 0..1')
        ordered = sorted(zip(rates, maps))
        if not all(left[1] < right[1] for left, right in zip(ordered, ordered[1:])):
            raise ValueError('BD-rate-mAP points must form a strictly increasing bitrate-mAP curve')
        return zip(*sorted((quality, math.log(rate)) for quality, rate in zip(maps, rates)))

    def fit_cubic(xs, ys):
        matrix = [[sum(x ** (row + column) for x in xs) for column in range(4)] +
                  [sum(y * x ** row for x, y in zip(xs, ys))] for row in range(4)]
        for column in range(4):
            pivot = max(range(column, 4), key=lambda row: abs(matrix[row][column]))
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            if abs(matrix[column][column]) < 1e-12:
                raise ValueError('mAP points cannot produce a stable BD-rate curve')
            divisor = matrix[column][column]
            matrix[column] = [value / divisor for value in matrix[column]]
            for row in range(4):
                if row == column:
                    continue
                factor = matrix[row][column]
                matrix[row] = [value - factor * pivot_value
                               for value, pivot_value in zip(matrix[row], matrix[column])]
        return [matrix[row][4] for row in range(4)]

    def integral(coefficients, lower, upper):
        return sum(coefficient * (upper ** (power + 1) - lower ** (power + 1)) / (power + 1)
                   for power, coefficient in enumerate(coefficients))

    anchor_map, anchor_rate = curve(anchor_points)
    test_map, test_rate = curve(test_points)
    anchor_map, anchor_rate = tuple(anchor_map), tuple(anchor_rate)
    test_map, test_rate = tuple(test_map), tuple(test_rate)
    lower = max(min(anchor_map), min(test_map))
    upper = min(max(anchor_map), max(test_map))
    if upper <= lower:
        raise ValueError('Anchor and test mAP ranges do not overlap')

    anchor_avg = integral(fit_cubic(anchor_map, anchor_rate), lower, upper)
    test_avg = integral(fit_cubic(test_map, test_rate), lower, upper)
    return (math.exp((test_avg - anchor_avg) / (upper - lower)) - 1) * 100


if __name__ == '__main__':
    anchor = [{'bitrate_kbps': rate, 'map50_95': quality}
              for rate, quality in zip((100, 200, 400, 800), (0.2, 0.3, 0.4, 0.5))]
    test = [{**point, 'bitrate_kbps': point['bitrate_kbps'] * 0.8} for point in anchor]
    assert abs(bd_rate_map(anchor, test) + 20) < 1e-6
    print('BD-rate-mAP self-check passed')
