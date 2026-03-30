#!/usr/bin/env python3
"""
Image Steganography Application
================================
Author      : [Your Name]
Reg ID      : [Your Reg ID]
Course      : Digital Image Processing
Instructor  : Mr. Ghulam Ali Khan
Description :
    Encode a secret image inside a cover image using 4-bit LSB steganography.
    Decode / recover the hidden image from a stego image.

Usage:
    # Encode (hide secret.png inside cover.png)
    python steganography_app.py --mode encode --cover cover.png --secret secret.png --output stego.png --show

    # Decode (extract hidden image from stego.png)
    python steganography_app.py --mode decode --stego stego.png --output recovered.png --show
"""

import argparse
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ─────────────────────────── CORE FUNCTIONS ───────────────────────────────

def load_and_match(cover_path: str, secret_path: str):
    """Load both images and resize secret to match cover dimensions."""
    cover  = cv2.imread(cover_path)
    secret = cv2.imread(secret_path)
    if cover is None:
        raise FileNotFoundError(f"Cover image not found: {cover_path}")
    if secret is None:
        raise FileNotFoundError(f"Secret image not found: {secret_path}")
    secret = cv2.resize(secret, (cover.shape[1], cover.shape[0]),
                        interpolation=cv2.INTER_LANCZOS4)
    return cover, secret


def encode(cover: np.ndarray, secret: np.ndarray) -> np.ndarray:
    """
    4-bit LSB Encoding.

    Formula:
        stego = (cover & 0xF0) | (secret >> 4)

    Explanation:
        - (cover  & 0xF0) : keeps top 4 bits of cover, zeroes bottom 4
        - (secret >> 4)   : shifts secret right → top 4 bits go to lower 4 positions
        - Bitwise OR      : merges them into one byte
    """
    return ((cover & 0xF0) | (secret >> 4)).astype(np.uint8)


def decode(stego: np.ndarray) -> np.ndarray:
    """
    4-bit LSB Decoding.

    Formula:
        recovered = (stego & 0x0F) << 4

    Explanation:
        - (stego & 0x0F) : isolates lower 4 bits (where secret is stored)
        - (<< 4)         : shifts them back to upper 4 bits → recovers 0-240 range
    """
    return ((stego & 0x0F) << 4).astype(np.uint8)


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio. Higher = more similar = better imperceptibility."""
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def show_comparison(images: list, titles: list, suptitle: str = "", save_path: str = None):
    """Display images side by side with coloured borders."""
    n = len(images)
    border_colors = ['#00bfff', '#ff6b6b', '#51cf66', '#ffd43b']
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.patch.set_facecolor('#0d1117')
    if n == 1:
        axes = [axes]
    for ax, img, title, bc in zip(axes, images, titles, border_colors):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color=bc, fontsize=11, fontweight='bold')
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(bc); sp.set_linewidth(3)
    if suptitle:
        fig.suptitle(suptitle, color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[✓] Result figure saved → {save_path}")
    plt.show()


# ─────────────────────────── MAIN CLI ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Image Steganography — Hide or Reveal a Secret Image",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  Encode:  python steganography_app.py --mode encode --cover cover.png --secret secret.png --output stego.png --show
  Decode:  python steganography_app.py --mode decode --stego stego.png --output recovered.png --show
        """
    )
    parser.add_argument('--mode',   choices=['encode', 'decode'], required=True,
                        help="'encode' to hide an image | 'decode' to reveal it")
    parser.add_argument('--cover',  help='Path to cover (host) image  [encode only]')
    parser.add_argument('--secret', help='Path to secret image        [encode only]')
    parser.add_argument('--stego',  help='Path to stego image         [decode only]')
    parser.add_argument('--output', required=True, help='Output file path (.png recommended)')
    parser.add_argument('--show',   action='store_true', help='Show result comparison plot')

    args = parser.parse_args()

    if args.mode == 'encode':
        if not args.cover or not args.secret:
            print("[ERROR] Encode mode requires --cover and --secret.")
            sys.exit(1)

        print(f"[•] Loading images …")
        cover, secret = load_and_match(args.cover, args.secret)

        print(f"[•] Encoding secret image into cover …")
        stego = encode(cover, secret)

        cv2.imwrite(args.output, stego)
        psnr_val = compute_psnr(cover, stego)

        print(f"[✓] Stego image saved    → {args.output}")
        print(f"    Cover size           : {cover.shape[1]}×{cover.shape[0]}")
        print(f"    PSNR (cover→stego)  : {psnr_val:.2f} dB  (>30 dB = imperceptible)")

        if args.show:
            show_comparison(
                [cover, secret, stego],
                ['Cover Image', 'Secret Image', f'Stego Image\nPSNR={psnr_val:.1f}dB'],
                suptitle='Steganography — Encode Result'
            )

    elif args.mode == 'decode':
        if not args.stego:
            print("[ERROR] Decode mode requires --stego.")
            sys.exit(1)

        print(f"[•] Loading stego image …")
        stego = cv2.imread(args.stego)
        if stego is None:
            raise FileNotFoundError(f"Stego image not found: {args.stego}")

        print(f"[•] Decoding hidden image …")
        recovered = decode(stego)

        cv2.imwrite(args.output, recovered)
        print(f"[✓] Recovered secret     → {args.output}")

        if args.show:
            show_comparison(
                [stego, recovered],
                ['Stego Image', 'Recovered Secret'],
                suptitle='Steganography — Decode Result'
            )


if __name__ == '__main__':
    main()
