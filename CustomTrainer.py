from ultralytics.models.yolo.segment import SegmentationTrainer
import cv2

class CustomTrainer(SegmentationTrainer):
    def preprocess_batch(self, batch):
        batch = cv2.cvtColor(batch['img'], cv2.COLOR_BGR2GRAY)
        batch = cv2.applyColorMap(batch['img'], cv2.COLORMAP_PARULA)

        return batch