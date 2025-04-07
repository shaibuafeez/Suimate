# Sui NFT Image Link Extractor

This repository contains scripts to extract image links from various NFT collections on the Sui blockchain using the BlockVision API.

## Collections

The following NFT collections are included:

1. **Killa Club** - Object type: `0x5fb75feb424b7fd3c7fd0d3cd3c5c9d9f03c8675a115c5c4e42af2c3a50b2f4e::killa_club::KillaClub`
   - Uses IPFS URLs for images
   - CSV file: `killaclub_links.csv`

2. **Kumo** - Object type: `0x5d0ced2b9a10153a68b0ba8d3db8bb347d64d70b0c3f6865c4b5d2d8b1d5a7a5::kumo::Kumo`
   - Uses IPFS URLs for images
   - CSV file: `kumo_1000_links.csv`

3. **Prime Machine** - Object type: `0x034c162f6b594cb5a1805264dd01ca5d80ce3eca6522e6ee37fd9ebfb9d3ddca::factory::PrimeMachin`
   - Uses SM.xyz URLs for images (`https://img.sm.xyz/{object_id}/`)
   - CSV file: `primemachine_1000_links.csv`

4. **Rootlet** - Object type: `0x8f74a7d632191e29956df3843404f22d27bd84d92cca1b1abde621d033098769::rootlet::Rootlet`
   - Uses base64-encoded data URLs for images
   - CSV file: `rootlet_1000_links.csv`

## Scripts

- `extract_killaclub_links.py` - Extracts Killa Club NFT image links
- `extract_kumo_links.py` - Extracts Kumo NFT image links
- `extract_primemachine_links.py` - Extracts Prime Machine NFT image links
- `extract_rootlet_links.py` - Extracts Rootlet NFT image links

## Usage

Each script can be run independently to extract 1000 NFT image links from its respective collection:

```bash
python extract_killaclub_links.py
python extract_kumo_links.py
python extract_primemachine_links.py
python extract_rootlet_links.py
```

The scripts will save the extracted links to CSV files with columns for:
- NFT name
- Object ID
- Image URl
