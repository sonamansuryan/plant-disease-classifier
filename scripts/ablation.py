import timm


CANDIDATES = [
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b4",
    "mobilenetv3_large_100",
    "resnet50",
    "convnext_tiny",
]

SELECTED = "efficientnet_b4"


def main():
    print(f"\n{'Backbone':<25} | {'Params (M)':>10} | {'Feature Dim':>11} | {'Note'}")
    print("-" * 70)
    for name in CANDIDATES:
        try:
            m      = timm.create_model(name, pretrained=False, num_classes=0, global_pool="avg")
            params = sum(x.numel() for x in m.parameters()) / 1e6
            feat   = m.num_features
            note   = "SELECTED" if name == SELECTED else ""
            print(f"{name:<25} | {params:>8.2f}M | {feat:>11} | {note}")
        except Exception as exc:
            print(f"{name:<25} | ERROR: {exc}")


if __name__ == "__main__":
    main()
