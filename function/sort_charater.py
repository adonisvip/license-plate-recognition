import numpy as np

def sort_by_rows(chars):
    if not chars:
        return ""

    heights = [y_max - y_min for (_, y_min, y_max), _ in chars]
    avg_height = np.mean(heights) * 0.6 

    rows = []
    current_row = []
    for char in sorted(chars, key=lambda c: c[0][1]): 
        if not current_row:
            current_row.append(char)
        else:
            prev_y_min = current_row[-1][0][1]
            if char[0][1] - prev_y_min > avg_height:
                rows.append(current_row)
                current_row = [char]
            else:
                current_row.append(char)
    
    if current_row:
        rows.append(current_row)

    # Sắp xếp từng hàng theo x_min (trái sang phải)
    # sorted_chars = []
    # for row in rows:
    #     sorted_chars.extend(sorted(row, key=lambda c: c[0][0]))  # Sắp xếp theo x_min

    # return "".join(c[1] for c in sorted_chars)

    plate_text_parts = []
    for row in rows:
        sorted_row = sorted(row, key=lambda c: c[0][0])
        plate_text_parts.append("".join(c[1] for c in sorted_row))

    return " ".join(plate_text_parts)