import supervisely as sly
import os
import csv
from collections import defaultdict
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.json import load_json_file
from supervisely.io.fs import get_file_name, get_file_name_with_ext
import shutil

from tqdm import tqdm

def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    
    dataset_path = os.path.join("all_fields_lincolnbeet","all")
    image_characteristics_path = os.path.join("all_fields_lincolnbeet","image_characteristics.csv")
    train_anns_path = os.path.join("all_fields_lincolnbeet","json_train_set.json")
    test_anns_path = os.path.join("all_fields_lincolnbeet","json_test_set.json")
    val_anns_path = os.path.join("all_fields_lincolnbeet","json_val_set.json")
    batch_size = 300

    ds_name_to_anns = {"train": train_anns_path, "val": val_anns_path, "test": test_anns_path}


    def create_ann(image_path):
        labels = []

        tags_data = im_name_to_tags[get_file_name_with_ext(image_path)]
        item_density_value = tags_data[0]
        item_density = sly.Tag(item_density_meta, value=item_density_value)

        bboxes_occupation_value = float(tags_data[1])
        bboxes_occupation = sly.Tag(bboxes_occupation_meta, value=bboxes_occupation_value)

        average_size_value = float(tags_data[2])
        average_size = sly.Tag(average_size_meta, value=average_size_value)

        average_occlusion_value = float(tags_data[3])
        average_occlusion = sly.Tag(average_occlusion_meta, value=average_occlusion_value)

        # image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_name_to_shape[get_file_name_with_ext(image_path)][0]  # image_np.shape[0]
        img_wight = image_name_to_shape[get_file_name_with_ext(image_path)][1]  # image_np.shape[1]

        ann_data = image_name_to_ann_data[get_file_name_with_ext(image_path)]
        for curr_ann_data in ann_data:
            category_id = curr_ann_data[0]

            bbox_coord = curr_ann_data[1]
            rectangle = sly.Rectangle(
                top=int(bbox_coord[1]),
                left=int(bbox_coord[0]),
                bottom=int(bbox_coord[1] + bbox_coord[3]),
                right=int(bbox_coord[0] + bbox_coord[2]),
            )
            label_rectangle = sly.Label(rectangle, idx_to_obj_class[category_id])
            labels.append(label_rectangle)

        return sly.Annotation(
            img_size=(img_height, img_wight),
            labels=labels,
            img_tags=[item_density, bboxes_occupation, average_size, average_occlusion],
        )


    sugar = sly.ObjClass("sugar_beet", sly.Rectangle)
    weed = sly.ObjClass("weed", sly.AnyGeometry)
    idx_to_obj_class = {0: sugar, 1: weed}

    item_density_meta = sly.TagMeta("item_density", sly.TagValueType.ANY_STRING)
    bboxes_occupation_meta = sly.TagMeta("bboxes_occupation", sly.TagValueType.ANY_NUMBER)
    average_size_meta = sly.TagMeta("average_relative_size", sly.TagValueType.ANY_NUMBER)
    average_occlusion_meta = sly.TagMeta("average_levels_occlusion", sly.TagValueType.ANY_NUMBER)
    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[sugar, weed],
        tag_metas=[
            item_density_meta,
            bboxes_occupation_meta,
            average_size_meta,
            average_occlusion_meta,
        ],
    )
    api.project.update_meta(project.id, meta.to_json())

    im_name_to_tags = {}
    with open(image_characteristics_path, "r") as file:
        csvreader = csv.reader(file)
        for idx, row in enumerate(csvreader):
            if idx == 0:
                continue
            im_name_to_tags[row[2]] = row[-4:]


    for ds_name, ann_path in ds_name_to_anns.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        ann = load_json_file(ann_path)

        image_id_to_name = {}
        image_name_to_shape = {}
        image_name_to_ann_data = defaultdict(list)

        for curr_image_info in ann["images"]:
            image_id_to_name[curr_image_info["id"]] = curr_image_info["file_name"]
            image_name_to_shape[curr_image_info["file_name"]] = (
                int(curr_image_info["height"]),
                int(curr_image_info["width"]),
            )

        for curr_ann_data in ann["annotations"]:
            image_id = curr_ann_data["image_id"]
            image_name_to_ann_data[image_id_to_name[image_id]].append(
                [curr_ann_data["category_id"], curr_ann_data["bbox"]]
            )

        images_names = list(image_name_to_ann_data.keys())

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(dataset_path, image_path) for image_path in img_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))

    return project
