import os
import json
from tqdm import tqdm
from PIL import Image, ImageDraw


CITIES = ["Chicago", "NewYork", "SanFrancisco", "Seattle"]


def arrange_sate_images(sate_image_names):
    lat_lons = [
        image_name.split(".png")[0].split("_")[1:3] for image_name in sate_image_names
    ]
    lat_lons = [[float(item) for item in lat_lon] for lat_lon in lat_lons]

    lats = [lat_lon[0] for lat_lon in lat_lons]
    lons = [lat_lon[1] for lat_lon in lat_lons]

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    def get_quadrant(lat, lon):
        if lat >= center_lat and lon >= center_lon:
            return 0
        elif lat >= center_lat and lon < center_lon:
            return 1
        elif lat < center_lat and lon < center_lon:
            return 2
        else:
            return 3

    points_with_quadrant = [
        (lat_lon, get_quadrant(lat_lon[0], lat_lon[1])) for lat_lon in lat_lons
    ]
    sorted_image_names = [""] * len(sate_image_names)
    for i, (_, index) in enumerate(points_with_quadrant):
        sorted_image_names[index] = sate_image_names[i]

    return sorted_image_names


def arrange_deltas(deltas):
    def get_delta_quadrant(x, y):
        if x >= 0 and y < 0:
            return 0
        elif x < 0 and y < 0:
            return 1
        elif x < 0 and y >= 0:
            return 2
        else:
            return 3

    deltas = [[float(item) for item in delta] for delta in deltas]
    deltas_with_quadrant = [
        (delta, get_delta_quadrant(delta[0], delta[1])) for delta in deltas
    ]
    deltas_with_quadrant = sorted(deltas_with_quadrant, key=lambda x: x[1])
    sorted_deltas = [item for item, _ in deltas_with_quadrant]

    return sorted_deltas


if __name__ == "__main__":

    vigor_root = "/home/anxiao/Datasets/MIGRANT/VIGOR"
    save_json_path = os.path.join(vigor_root, "cross_view_grounding.json")
    json_data = []
    for city in CITIES:
        satellite_dir = os.path.join(vigor_root, city, "satellite")
        panorama_dir = os.path.join(vigor_root, city, "panorama")
        label_dir = os.path.join(vigor_root, "splits", city)
        txt_file = os.path.join(label_dir, "same_area_balanced_train.txt")
        with open(txt_file, "r") as f:
            lines = f.readlines()
        for line_number, line in enumerate(tqdm(lines), start=1):
            contents = line.strip().split(" ")
            pano_name = contents[0]
            sate_names = contents[1:12:3]
            deltas = [contents[2:4], contents[5:7], contents[8:10], contents[11:13]]

            sate_names = arrange_sate_images(sate_names)
            deltas = arrange_deltas(deltas)

            if "semi" not in save_json_path:
                abs_deltas = [[abs(float(item)) for item in delta] for delta in deltas]
                sate_paths = [
                    os.path.join(satellite_dir, sate_name) for sate_name in sate_names
                ]
                sate_sizes = [Image.open(sate_path).size for sate_path in sate_paths]
                for i, abs_delta in enumerate(abs_deltas):
                    if (
                        abs_delta[0] <= sate_sizes[i][0] / 4
                        and abs_delta[1] <= sate_sizes[i][1] / 4
                    ):
                        break
                point_coord = [
                    sate_sizes[i][0] / 2 - deltas[i][0],
                    sate_sizes[i][1] / 2 - deltas[i][1],
                ]
                pano_path = os.path.join(panorama_dir, pano_name)

                sate_image = Image.open(sate_paths[i])
                plot_dir = os.path.join(vigor_root, "plot_pos", city)
                os.makedirs(plot_dir, exist_ok=True)
                draw = ImageDraw.Draw(sate_image)
                draw.ellipse((point_coord[0] - 5, point_coord[1] - 5, point_coord[0] + 5, point_coord[1] + 5), fill="red")
                sate_image.save(os.path.join(plot_dir, f"{city}_{line_number}.png"))

                json_data.append(
                    {
                        "pano_path": pano_path,
                        "sate_path": sate_paths[i],
                        "sate_plot_path": os.path.join(plot_dir, f"{city}_{line_number}.png"),
                        "point_coords": point_coord,
                    }
                )
            else:
                pano_path = os.path.join(panorama_dir, pano_name)
                sate_paths = [
                    os.path.join(satellite_dir, sate_name) for sate_name in sate_names
                ]
                sate_sizes = [Image.open(sate_path).size for sate_path in sate_paths]
                point_coords = [
                    (sate_size[0] / 2 - delta[0], sate_size[1] / 2 - delta[1])
                    for sate_size, delta in zip(sate_sizes, deltas)
                ]
                sate_images = [Image.open(sate_path) for sate_path in sate_paths]

                plot_dir = os.path.join(vigor_root, "plot_pos_semi", city)
                os.makedirs(plot_dir, exist_ok=True)
                for i, sate_image in enumerate(sate_images):
                    draw = ImageDraw.Draw(sate_image)
                    point_coord = point_coords[i]
                    draw.ellipse(
                        (
                            point_coord[0] - 5,
                            point_coord[1] - 5,
                            point_coord[0] + 5,
                            point_coord[1] + 5,
                        ),
                        fill="red",
                    )
                    sate_image.save(os.path.join(plot_dir, f"{city}_{line_number}_{i+1}.png"))

                json_data.append(
                    {
                        "pano_path": pano_path,
                        "sate_paths": sate_paths,
                        "sate_plot_paths": [os.path.join(plot_dir, f"{line_number}_{i+1}.png") for i in range(4)],
                        "point_coords": point_coords,
                    }
                )

    with open(save_json_path, "w") as f:
        json.dump(json_data, f, indent=4)
