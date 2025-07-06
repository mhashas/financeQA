from PIL import Image
from ultralyticsplus import YOLO

from financeqa.preprocessing.data_models import DetectionResult


class TableDetector:

    def __init__(self):
        self.model = YOLO("foduucom/table-detection-and-extraction")

        # set model parameters
        self.model.overrides["conf"] = 0.25  # NMS confidence threshold
        self.model.overrides["iou"] = 0.45  # NMS IoU threshold
        self.model.overrides["agnostic_nms"] = False  # NMS class-agnostic
        self.model.overrides["max_det"] = 1000  # maximum number of detections per image

    def detect(
        self,
        image: Image.Image,
        *,
        top_ext_perc: float = 0.15,
        bottom_ext_perc: float = 0.15,
        left_ext_perc: float = 0.15,
        right_ext_perc: float = 0.15
    ) -> list[DetectionResult]:
        """
        Detect tables in an image

        Args:
            image: image in which to detect tables
            top_ext_perc: percentage of the bounding box height to extend the top side
            bottom_ext_perc: percentage of the bounding box height to extend the bottom side
            left_ext_perc: percentage of the bounding box height to extend the left side
            right_ext_perc: percentage of the bounding box height to extend the right side

        Returns:
            cropped images containing detected tables and their bounding boxes
        """
        predictions = self.model.predict(image)
        bboxes = predictions[0].boxes

        if len(bboxes) == 0:
            return []

        tables = []
        for bbox in bboxes:
            pil_bbox = bbox.xyxy.tolist()[0]

            bbox_height = pil_bbox[3] - pil_bbox[1]
            extended_x0 = max(0, pil_bbox[0] - left_ext_perc * bbox_height)
            extended_y0 = max(0, pil_bbox[1] - top_ext_perc * bbox_height)
            extended_x1 = min(image.width, pil_bbox[2] + right_ext_perc * bbox_height)
            extended_y1 = min(image.height, pil_bbox[3] + bottom_ext_perc * bbox_height)

            table_image = image.crop((extended_x0, extended_y0, extended_x1, extended_y1))
            result = DetectionResult(image=table_image, bbox=pil_bbox)
            tables.append(result)

        return tables
