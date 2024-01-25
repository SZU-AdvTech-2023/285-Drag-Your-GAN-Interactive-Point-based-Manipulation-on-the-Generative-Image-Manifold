from PySide6.QtWidgets import QApplication, QWidget, QLabel, QMessageBox
from PySide6.QtGui import QPainter, QPixmap, QImage
from PySide6.QtCore import QSize, QPoint
from components.ImageLabel import ImageLabel
from PIL import Image, ImageQt
import os

from pprint import pprint

class ImageWidget(QWidget):
    """
    图片自适应 QWidget (通过QLabel显式)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.BASE_SIZE = 512

        self.image_label = ImageLabel(self)
        self.image_label.setScaledContents(True)
        self.image_label.installEventFilter(self)
        self.image_rate = None
        self.last_pos = None
        self.array_img = None
        self.pixmap = QPixmap()

        self.painter = QPainter()

    def set_image(self, file_name, base_size=512):
        try:
            self.BASE_SIZE = base_size
            self.pixmap = QPixmap().fromImage(QImage(file_name))
            pix_map = self.pixmap
            self.image_rate = pix_map.width() / pix_map.height()
            self.image_label.setPixmap(pix_map)
            self.compute_size()
        except Exception as e:
            QMessageBox.critical(self, "Error", "Load image failed!")
            print(e)

    def set_image_from_array(self, image, base_size=512):
        self.BASE_SIZE = base_size
        self.array_img = image.cpu().numpy()
        img = Image.fromarray(self.array_img)
        self.pixmap = img.toqpixmap()
        pix_map = self.pixmap
        self.image_rate = pix_map.width() / pix_map.height()
        self.image_label.setPixmap(pix_map)
        self.compute_size()

    def get_image(self):
        if self.array_img is not None:
            return self.array_img
        return None

    def compute_size(self):
        if self.image_rate is not None:
            w = self.size().width()
            h = self.size().height()
            scale_w = int(h * self.image_rate)

            scale = 1
            if scale_w <= w:
                self.image_label.resize(QSize(scale_w, h))
                self.image_label.setProperty(
                    "pos", QPoint(int((w - scale_w) / 2), 0))
                scale = scale_w/self.BASE_SIZE
            else:
                scale_h = int(w / self.image_rate)
                self.image_label.resize(QSize(w, scale_h))
                self.image_label.setProperty(
                    "pos", QPoint(0, int((h - scale_h) / 2)))
                scale = scale_h/self.BASE_SIZE
            self.set_image_scale(scale)
        else:
            self.image_label.resize(self.size())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.compute_size()

    def save_image(self, filename, format, quality, is_experience=False):
        """
        pixmap.save()方法是用于将QPixmap对象保存为文件的方法，它可以写入多种不同的文件格式，具体支持的格式取决于Qt库和操作系统的支持情况。
        在大多数情况下，Qt库支持的格式包括：
        1. BMP格式：使用"bmp"或"wbmp"文件扩展名
        2. GIF格式：使用"gif"文件扩展名
        3. JPEG格式：使用"jpg"、"jpeg"或"jpe"文件扩展名
        4. PNG格式：使用"png"文件扩展名
        5. PBM格式：使用"pbm"文件扩展名
        6. PGM格式：使用"pgm"文件扩展名
        7. PPM格式：使用"ppm"文件扩展名
        8. XBM格式：使用"xbm"文件扩展名
        9. XPM格式：使用"xpm"文件扩展名
        """
        short_name = os.path.basename(filename)
        if self.pixmap:
            if self.pixmap.save(filename, format, quality):
                if is_experience:
                    print(f"Image Saved as {short_name} successfully!")
                else:
                    QMessageBox.information(self, "Save Image", f"Image saved as {short_name} successfully!")
            elif is_experience:
                print(f"Save image as {short_name} failed!")
            else:
                QMessageBox.critical(self, "Error", f"Save image as {short_name} failed!")
        elif is_experience:
            print("No image loaded!")
        else:
            QMessageBox.critical(self, "Error", "No image loaded!")

    def get_image_scale(self):
        return self.image_label.getImageScale()

    def set_image_scale(self, image_scale):
        self.image_label.setImageScale(image_scale)

    def set_status(self, new_status):
        self.image_label.set_status(new_status)

    def get_points(self):
        return self.image_label.get_points()

    def set_points(self, points):
        self.image_label.set_points(points)

    def add_points(self, points):
        self.image_label.add_points(points)

    def clear_points(self):
        self.image_label.clear_points()

if __name__ == "__main__":
    app = QApplication([])
    # widget = QWidget()
    # widget.show()
    image = ImageWidget()
    image.set_image(os.path.realpath("./components/dog.jpg"))
    image.show()
    app.exec()
