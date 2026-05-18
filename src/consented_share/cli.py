import argparse

from consented_share.pipeline import run_consented_share_pipeline


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-root", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-image", default=None)
    parser.add_argument("--placement-mode", choices=["bbox", "mask"], default="mask")
    parser.add_argument("--private-key", default="../keys/device_private.pem", help="Path to device RSA public key PEM file")

    args = parser.parse_args()

    run_consented_share_pipeline(
        output_root=args.output_root,
        private_key_path=args.private_key,
        request_text=args.request,
        placement_mode=args.placement_mode,
        output_image_path=args.output_image,
    )


if __name__ == "__main__":
    main()

#python3 -m consented_share.cli --output-root ../data/output/predict_10  --request "ceraVe Cream Bottle"
# python3 -m consented_share.cli --output-root ../data/output/predict_10  --request "Coffe Table"