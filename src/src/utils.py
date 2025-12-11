def group_boxes_by_row(boxes, threshold=20):
    rows = []
    boxes = sorted(boxes, key=lambda b: b[1])

    for box in boxes:
        cy = (box[1] + box[3]) / 2
        placed = False

        for row in rows:
            if abs(cy - row["cy"]) < threshold:
                row["boxes"].append(box)
                row["cy"] = (row["cy"] * (len(row["boxes"]) - 1) + cy) / len(row["boxes"])
                placed = True
                break

        if not placed:
            rows.append({"cy": cy, "boxes": [box]})

    return rows
