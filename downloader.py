import os

# import sys
# sys.path.append(r'C:\Users\krits\Downloads\testSpeedTracking\supervision\examples\speed_estimation\venv\Lib\site-packages')

from supervision.assets import VideoAssets, download_assets

if not os.path.exists("data"):
    os.makedirs("data")
os.chdir("data")
download_assets(VideoAssets.VEHICLES)
