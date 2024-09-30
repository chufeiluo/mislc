import torch

def get_spans(mask, value):
    start = None
    stop = None
    spans = []
    for i in range(len(mask) - 1):
        if mask[i] == value and start is None:
            start = i
            stop = i + 1
        if start is not None:
            if mask[stop] == value:
                stop += 1
            else:  # span ends
                spans.append((start, stop))
                start = None
    if start is not None:
        spans.append((start, len(mask)))

    return spans

a = torch.Tensor([0, 0, 1, 1, 1, 0, 1, 1])
spans = get_spans(a, 1)

print(spans)
print(a[spans[0][0]:spans[0][1]])
