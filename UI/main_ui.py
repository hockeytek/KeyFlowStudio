# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton, QSizePolicy,
    QSlider, QSpacerItem, QSpinBox, QStatusBar,
    QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1660, 930)
        MainWindow.setMinimumSize(QSize(1660, 930))
        MainWindow.setStyleSheet(u"QMainWindow {\n"
"    background-color: #0e1014;\n"
"}\n"
"QWidget {\n"
"    background-color: #11141a;\n"
"    color: #e8edf5;\n"
"    font-family: \"Avenir Next\", \"Helvetica Neue\", \"Arial\", sans-serif;\n"
"    font-size: 13px;\n"
"}\n"
"QPushButton {\n"
"    background-color: #1b212b;\n"
"    border: 1px solid #2a3444;\n"
"    color: #eef3fb;\n"
"    padding: 6px 12px;\n"
"    border-radius: 9px;\n"
"    font-weight: 600;\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: #243145;\n"
"    border: 1px solid #43c7ff;\n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: #122235;\n"
"    border: 1px solid #21b8ff;\n"
"}\n"
"QPushButton:checked {\n"
"    background-color: #0f3c57;\n"
"    border: 1px solid #34c2ff;\n"
"    color: #f7fbff;\n"
"}\n"
"QPushButton:disabled {\n"
"    background-color: #131820;\n"
"    border: 1px solid #202835;\n"
"    color: #5b6473;\n"
"}\n"
"QComboBox {\n"
"    background-color: #171d27;\n"
"    border: 1px solid #2a3444;\n"
"    color: #e8edf5;\n"
"    padding: 4px "
                        "6px;\n"
"    border-radius: 8px;\n"
"    min-height: 28px;\n"
"}\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: top right;\n"
"    min-width: 0px;\n"
"    max-width: 0px;\n"
"    width: 0px;\n"
"    border: none;\n"
"    margin: 0px;\n"
"    padding: 0px;\n"
"}\n"
"QComboBox::down-arrow {\n"
"    image: none;\n"
"    width: 0px;\n"
"    height: 0px;\n"
"}\n"
"QComboBox:hover {\n"
"    border: 1px solid #43c7ff;\n"
"}\n"
"QComboBox QAbstractItemView {\n"
"    background-color: #131a23;\n"
"    color: #e8edf5;\n"
"    selection-background-color: #0f3c57;\n"
"    border: 1px solid #2a3444;\n"
"}\n"
"QSlider::groove:horizontal {\n"
"    height: 7px;\n"
"    background: #222c3a;\n"
"    border-radius: 3px;\n"
"}\n"
"QSlider::sub-page:horizontal {\n"
"    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #28b8ff, stop:1 #5fd7ff);\n"
"    border-radius: 3px;\n"
"}\n"
"QSlider::handle:horizontal {\n"
"    background: #eaf7ff;\n"
"    border: 2px solid #24bbff;\n"
""
                        "    width: 16px;\n"
"    height: 16px;\n"
"    margin: -6px 0;\n"
"    border-radius: 8px;\n"
"}\n"
"QProgressBar {\n"
"    background-color: #151c26;\n"
"    border: 1px solid #2a3444;\n"
"    border-radius: 7px;\n"
"    text-align: center;\n"
"    color: #d7e9f7;\n"
"    font-size: 11px;\n"
"    font-weight: 600;\n"
"}\n"
"QProgressBar::chunk {\n"
"    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1fa6ff, stop:1 #64d9ff);\n"
"    border-radius: 6px;\n"
"}\n"
"QSpinBox {\n"
"    background-color: #171d27;\n"
"    border: 1px solid #2a3444;\n"
"    color: #e8edf5;\n"
"    padding: 3px 8px;\n"
"    border-radius: 7px;\n"
"    min-height: 25px;\n"
"}\n"
"QCheckBox {\n"
"    color: #cfd7e3;\n"
"    font-size: 12px;\n"
"    spacing: 6px;\n"
"}\n"
"QCheckBox::indicator {\n"
"    width: 12px;\n"
"    height: 12px;\n"
"    border-radius: 3px;\n"
"    border: 1px solid #3a4657;\n"
"    background-color: #171d27;\n"
"}\n"
"QCheckBox::indicator:checked {\n"
"    background-color: #24bbff;\n"
"    border: 1"
                        "px solid #24bbff;\n"
"}\n"
"QFrame#line1, QFrame#line2 {\n"
"    color: #263244;\n"
"    background-color: #263244;\n"
"    max-height: 1px;\n"
"}\n"
"QLabel#lbl_brand {\n"
"    color: #d8f4ff;\n"
"    font-size: 14px;\n"
"    font-weight: 700;\n"
"    letter-spacing: 1px;\n"
"    padding: 0 6px;\n"
"}\n"
"QLabel#lbl_brand_badge {\n"
"    color: #8ce1ff;\n"
"    background-color: #16364a;\n"
"    border: 1px solid #27546e;\n"
"    border-radius: 8px;\n"
"    font-size: 10px;\n"
"    font-weight: 700;\n"
"    padding: 3px 8px;\n"
"}\n"
"QLabel#lbl_input_title,\n"
"QLabel#lbl_output_title,\n"
"QLabel#lbl_sec_params,\n"
"QLabel#lbl_sec_actions {\n"
"    color: #8ca0ba;\n"
"}\n"
"QLabel#input_video_label,\n"
"QLabel#output_video_label {\n"
"    background-color: #0b0f15;\n"
"    border: 1px solid #2a3444;\n"
"    border-radius: 12px;\n"
"    color: #6d7f95;\n"
"    font-size: 14px;\n"
"}\n"
"QLabel#input_video_label:hover,\n"
"QLabel#output_video_label:hover {\n"
"    border: 1px solid #43c7ff;\n"
"}\n"
"QPushButt"
                        "on#btn_play {\n"
"    font-size: 16px;\n"
"    font-weight: 700;\n"
"    color: #e8f7ff;\n"
"    min-width: 39px;\n"
"    max-width: 39px;\n"
"    min-height: 39px;\n"
"    max-height: 39px;\n"
"    border-radius: 6px;\n"
"    padding: 0;\n"
"    background-color: transparent;\n"
"    border: none;\n"
"}\n"
"QPushButton#btn_first_frame,\n"
"QPushButton#btn_prev_frame,\n"
"QPushButton#btn_next_frame,\n"
"QPushButton#btn_last_frame,\n"
"QPushButton#btn_play_loop {\n"
"    min-width: 32px;\n"
"    max-width: 32px;\n"
"    min-height: 32px;\n"
"    max-height: 32px;\n"
"    border-radius: 6px;\n"
"    font-size: 20px;\n"
"    font-weight: 700;\n"
"    padding: 0;\n"
"    background-color: transparent;\n"
"    border: none;\n"
"    color: #f2f7ff;\n"
"}\n"
"QPushButton#btn_play_reverse {\n"
"    min-width: 39px;\n"
"    max-width: 39px;\n"
"    min-height: 39px;\n"
"    max-height: 39px;\n"
"    border-radius: 6px;\n"
"    font-size: 20px;\n"
"    font-weight: 700;\n"
"    padding: 0;\n"
"    background-color: tran"
                        "sparent;\n"
"    border: none;\n"
"    color: #f2f7ff;\n"
"}\n"
"QPushButton#btn_first_frame:hover,\n"
"QPushButton#btn_prev_frame:hover,\n"
"QPushButton#btn_play_reverse:hover,\n"
"QPushButton#btn_play:hover,\n"
"QPushButton#btn_next_frame:hover,\n"
"QPushButton#btn_last_frame:hover,\n"
"QPushButton#btn_play_loop:hover {\n"
"    color: #ffffff;\n"
"}\n"
"QPushButton#btn_first_frame:pressed,\n"
"QPushButton#btn_prev_frame:pressed,\n"
"QPushButton#btn_play_reverse:pressed,\n"
"QPushButton#btn_play:pressed,\n"
"QPushButton#btn_next_frame:pressed,\n"
"QPushButton#btn_last_frame:pressed,\n"
"QPushButton#btn_play_loop:pressed {\n"
"    color: #9ad7ff;\n"
"}\n"
"QSlider#frame_slider {\n"
"    min-height: 12px;\n"
"}\n"
"QSlider#frame_slider::groove:horizontal {\n"
"    border: none;\n"
"    height: 2px;\n"
"    border-radius: 1px;\n"
"    background: #42526b;\n"
"}\n"
"QSlider#frame_slider::sub-page:horizontal {\n"
"    border-radius: 1px;\n"
"    background: #95a7bf;\n"
"}\n"
"QSlider#frame_slider::add-page:horizon"
                        "tal {\n"
"    border-radius: 1px;\n"
"    background: #42526b;\n"
"}\n"
"QSlider#frame_slider::handle:horizontal {\n"
"    width: 6px;\n"
"    margin: -8px 0;\n"
"    border-radius: 2px;\n"
"    background: #f4f7fb;\n"
"    border: 1px solid #d2dde8;\n"
"}\n"
"QLabel#lbl_frame_info {\n"
"    color: #f2f9ff;\n"
"    font-size: 28px;\n"
"    font-weight: 600;\n"
"    letter-spacing: 1px;\n"
"}\n"
"QLabel#lbl_playback_title {\n"
"    color: #8ca0ba;\n"
"    font-size: 10px;\n"
"    font-weight: 700;\n"
"    letter-spacing: 1px;\n"
"}\n"
"QPushButton#btn_play:checked,\n"
"QPushButton#btn_play_reverse:checked,\n"
"QPushButton#btn_play_loop:checked {\n"
"    background-color: #0f3c57;\n"
"    border: 1px solid #34c2ff;\n"
"    color: #d9f6ff;\n"
"}")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.vl_main = QVBoxLayout(self.centralwidget)
        self.vl_main.setSpacing(4)
        self.vl_main.setObjectName(u"vl_main")
        self.vl_main.setContentsMargins(6, 10, 6, 8)
        self.hl_topbar = QHBoxLayout()
        self.hl_topbar.setSpacing(6)
        self.hl_topbar.setObjectName(u"hl_topbar")
        self.lbl_brand = QLabel(self.centralwidget)
        self.lbl_brand.setObjectName(u"lbl_brand")
        self.lbl_brand.setMinimumSize(QSize(160, 34))
        self.lbl_brand.setMaximumSize(QSize(220, 34))

        self.hl_topbar.addWidget(self.lbl_brand)

        self.lbl_brand_badge = QLabel(self.centralwidget)
        self.lbl_brand_badge.setObjectName(u"lbl_brand_badge")
        self.lbl_brand_badge.setMinimumSize(QSize(60, 24))
        self.lbl_brand_badge.setMaximumSize(QSize(80, 24))
        self.lbl_brand_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.hl_topbar.addWidget(self.lbl_brand_badge)

        self.horizontalSpacer_3 = QSpacerItem(72, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.hl_topbar.addItem(self.horizontalSpacer_3)

        self.sp_file_label_left_pad = QSpacerItem(10, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.hl_topbar.addItem(self.sp_file_label_left_pad)

        self.lbl_file_path = QLabel(self.centralwidget)
        self.lbl_file_path.setObjectName(u"lbl_file_path")
        self.lbl_file_path.setStyleSheet(u"color: #8ce1ff; font-size: 12px; font-weight: 600;")
        self.lbl_file_path.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.hl_topbar.addWidget(self.lbl_file_path)

        self.sp_topbar_r = QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hl_topbar.addItem(self.sp_topbar_r)

        self.btn_settings = QPushButton(self.centralwidget)
        self.btn_settings.setObjectName(u"btn_settings")
        self.btn_settings.setMinimumSize(QSize(38, 38))
        self.btn_settings.setMaximumSize(QSize(38, 38))
        icon = QIcon()
        icon.addFile(u"app/assets/settings-gear.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btn_settings.setIcon(icon)
        self.btn_settings.setIconSize(QSize(18, 18))

        self.hl_topbar.addWidget(self.btn_settings)

        self.sp_topbar_device_right_pad = QSpacerItem(15, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.hl_topbar.addItem(self.sp_topbar_device_right_pad)


        self.vl_main.addLayout(self.hl_topbar)

        self.hl_viewers = QHBoxLayout()
        self.hl_viewers.setSpacing(2)
        self.hl_viewers.setObjectName(u"hl_viewers")
        self.vl_input = QVBoxLayout()
        self.vl_input.setSpacing(4)
        self.vl_input.setObjectName(u"vl_input")
        self.lbl_input_title = QLabel(self.centralwidget)
        self.lbl_input_title.setObjectName(u"lbl_input_title")
        self.lbl_input_title.setStyleSheet(u"color: #8ca0ba; font-size: 11px; font-weight: 700; letter-spacing: 1px;")
        self.lbl_input_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.vl_input.addWidget(self.lbl_input_title)

        self.input_video_label = QLabel(self.centralwidget)
        self.input_video_label.setObjectName(u"input_video_label")
        self.input_video_label.setMinimumSize(QSize(720, 405))
        self.input_video_label.setMaximumSize(QSize(16777215, 405))
        self.input_video_label.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.input_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.vl_input.addWidget(self.input_video_label)


        self.hl_viewers.addLayout(self.vl_input)

        self.vl_output = QVBoxLayout()
        self.vl_output.setSpacing(4)
        self.vl_output.setObjectName(u"vl_output")
        self.hl_output_header_row = QHBoxLayout()
        self.hl_output_header_row.setSpacing(6)
        self.hl_output_header_row.setObjectName(u"hl_output_header_row")
        self.sp_output_preview_left_pad = QSpacerItem(15, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.hl_output_header_row.addItem(self.sp_output_preview_left_pad)

        self.btn_preview_foreground = QPushButton(self.centralwidget)
        self.btn_preview_foreground.setObjectName(u"btn_preview_foreground")
        self.btn_preview_foreground.setEnabled(False)
        self.btn_preview_foreground.setMinimumSize(QSize(26, 26))
        self.btn_preview_foreground.setMaximumSize(QSize(26, 26))
        icon1 = QIcon()
        icon1.addFile(u"app/assets/preview-gamma-display.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btn_preview_foreground.setIcon(icon1)
        self.btn_preview_foreground.setIconSize(QSize(14, 14))
        self.btn_preview_foreground.setCheckable(True)
        self.btn_preview_foreground.setChecked(True)

        self.hl_output_header_row.addWidget(self.btn_preview_foreground)

        self.btn_preview_alpha = QPushButton(self.centralwidget)
        self.btn_preview_alpha.setObjectName(u"btn_preview_alpha")
        self.btn_preview_alpha.setEnabled(False)
        self.btn_preview_alpha.setMinimumSize(QSize(26, 26))
        self.btn_preview_alpha.setMaximumSize(QSize(26, 26))
        icon2 = QIcon()
        icon2.addFile(u"app/assets/preview-gamma-linear.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btn_preview_alpha.setIcon(icon2)
        self.btn_preview_alpha.setIconSize(QSize(14, 14))
        self.btn_preview_alpha.setCheckable(True)

        self.hl_output_header_row.addWidget(self.btn_preview_alpha)

        self.sp_output_title_left = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hl_output_header_row.addItem(self.sp_output_title_left)

        self.lbl_output_title = QLabel(self.centralwidget)
        self.lbl_output_title.setObjectName(u"lbl_output_title")
        self.lbl_output_title.setStyleSheet(u"color: #8ca0ba; font-size: 11px; font-weight: 700; letter-spacing: 1px;")
        self.lbl_output_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.hl_output_header_row.addWidget(self.lbl_output_title)

        self.sp_output_title_right = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hl_output_header_row.addItem(self.sp_output_title_right)

        self.btn_split_view = QPushButton(self.centralwidget)
        self.btn_split_view.setObjectName(u"btn_split_view")
        self.btn_split_view.setEnabled(False)
        self.btn_split_view.setMinimumSize(QSize(26, 26))
        self.btn_split_view.setMaximumSize(QSize(26, 26))
        self.btn_split_view.setCheckable(True)

        self.hl_output_header_row.addWidget(self.btn_split_view)

        self.sp_output_split_right_pad = QSpacerItem(15, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.hl_output_header_row.addItem(self.sp_output_split_right_pad)


        self.vl_output.addLayout(self.hl_output_header_row)

        self.output_video_label = QLabel(self.centralwidget)
        self.output_video_label.setObjectName(u"output_video_label")
        self.output_video_label.setMinimumSize(QSize(720, 405))
        self.output_video_label.setMaximumSize(QSize(16777215, 405))
        self.output_video_label.setStyleSheet(u"background-color: #101722;\n"
"background-image: qlineargradient(x1:0,y1:0,x2:1,y2:1,\n"
"    stop:0 #182231,\n"
"    stop:0.25 #182231,\n"
"    stop:0.25 #111a27,\n"
"    stop:0.5 #111a27,\n"
"    stop:0.5 #182231,\n"
"    stop:0.75 #182231,\n"
"    stop:0.75 #111a27,\n"
"    stop:1 #111a27);\n"
"border: 1px solid #2d3a4d;\n"
"border-radius: 12px;\n"
"color: #86a0bb;\n"
"font-size: 14px;")
        self.output_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.vl_output.addWidget(self.output_video_label)


        self.hl_viewers.addLayout(self.vl_output)


        self.vl_main.addLayout(self.hl_viewers)

        self.sp_between_viewers_and_controls = QSpacerItem(20, 12, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.vl_main.addItem(self.sp_between_viewers_and_controls)

        self.hl_control_row = QHBoxLayout()
        self.hl_control_row.setSpacing(8)
        self.hl_control_row.setObjectName(u"hl_control_row")
        self.hl_control_row.setContentsMargins(0, -1, 0, -1)
        self.vl_center_work = QVBoxLayout()
        self.vl_center_work.setSpacing(4)
        self.vl_center_work.setObjectName(u"vl_center_work")
        self.vl_center_work.setContentsMargins(15, 0, 15, 0)
        self.hl_slider_row = QHBoxLayout()
        self.hl_slider_row.setSpacing(0)
        self.hl_slider_row.setObjectName(u"hl_slider_row")
        self.frame_slider = QSlider(self.centralwidget)
        self.frame_slider.setObjectName(u"frame_slider")
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setOrientation(Qt.Orientation.Horizontal)

        self.hl_slider_row.addWidget(self.frame_slider)

        self.lbl_frame_info = QLabel(self.centralwidget)
        self.lbl_frame_info.setObjectName(u"lbl_frame_info")
        self.lbl_frame_info.setMinimumSize(QSize(170, 0))
        self.lbl_frame_info.setStyleSheet(u"font-size: 18px; color: #f2f9ff; font-weight: 600; letter-spacing: 1px;")
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.hl_slider_row.addWidget(self.lbl_frame_info)


        self.vl_center_work.addLayout(self.hl_slider_row)

        self.hl_play_row = QHBoxLayout()
        self.hl_play_row.setSpacing(8)
        self.hl_play_row.setObjectName(u"hl_play_row")
        self.hl_play_left_zone = QHBoxLayout()
        self.hl_play_left_zone.setSpacing(8)
        self.hl_play_left_zone.setObjectName(u"hl_play_left_zone")
        self.hl_play_left_zone.setContentsMargins(0, 0, 0, 0)
        self.combo_playback_presets = QComboBox(self.centralwidget)
        self.combo_playback_presets.addItem("")
        self.combo_playback_presets.setObjectName(u"combo_playback_presets")
        self.combo_playback_presets.setMinimumSize(QSize(210, 38))
        self.combo_playback_presets.setMaximumSize(QSize(250, 33))

        self.hl_play_left_zone.addWidget(self.combo_playback_presets)

        self.hl_n_frames = QHBoxLayout()
        self.hl_n_frames.setSpacing(4)
        self.hl_n_frames.setObjectName(u"hl_n_frames")
        self.hl_n_frames.setContentsMargins(0, 0, 0, 0)
        self.lbl_start_frame = QLabel(self.centralwidget)
        self.lbl_start_frame.setObjectName(u"lbl_start_frame")
        self.lbl_start_frame.setStyleSheet(u"font-size: 12px; color: #a9bfd4; font-weight: 600;")

        self.hl_n_frames.addWidget(self.lbl_start_frame)

        self.spin_start_frame = QSpinBox(self.centralwidget)
        self.spin_start_frame.setObjectName(u"spin_start_frame")
        self.spin_start_frame.setMinimumSize(QSize(72, 33))
        self.spin_start_frame.setMaximumSize(QSize(72, 33))
        self.spin_start_frame.setMinimum(0)
        self.spin_start_frame.setMaximum(99999)
        self.spin_start_frame.setValue(0)

        self.hl_n_frames.addWidget(self.spin_start_frame)

        self.lbl_end_frame = QLabel(self.centralwidget)
        self.lbl_end_frame.setObjectName(u"lbl_end_frame")
        self.lbl_end_frame.setStyleSheet(u"font-size: 12px; color: #a9bfd4; font-weight: 600;")

        self.hl_n_frames.addWidget(self.lbl_end_frame)

        self.spin_end_frame = QSpinBox(self.centralwidget)
        self.spin_end_frame.setObjectName(u"spin_end_frame")
        self.spin_end_frame.setMinimumSize(QSize(90, 33))
        self.spin_end_frame.setMaximumSize(QSize(90, 33))
        self.spin_end_frame.setMinimum(-1)
        self.spin_end_frame.setMaximum(99999)
        self.spin_end_frame.setValue(-1)

        self.hl_n_frames.addWidget(self.spin_end_frame)

        self.lbl_num_frames = QLabel(self.centralwidget)
        self.lbl_num_frames.setObjectName(u"lbl_num_frames")
        self.lbl_num_frames.setStyleSheet(u"font-size: 12px; color: #a9bfd4; font-weight: 600;")

        self.hl_n_frames.addWidget(self.lbl_num_frames)

        self.spin_num_frames = QSpinBox(self.centralwidget)
        self.spin_num_frames.setObjectName(u"spin_num_frames")
        self.spin_num_frames.setMinimumSize(QSize(72, 33))
        self.spin_num_frames.setMaximumSize(QSize(72, 33))
        self.spin_num_frames.setMinimum(0)
        self.spin_num_frames.setMaximum(99999)
        self.spin_num_frames.setValue(0)

        self.hl_n_frames.addWidget(self.spin_num_frames)


        self.hl_play_left_zone.addLayout(self.hl_n_frames)

        self.spacer_play_left_push = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hl_play_left_zone.addItem(self.spacer_play_left_push)


        self.hl_play_row.addLayout(self.hl_play_left_zone)

        self.hl_play_center_zone = QHBoxLayout()
        self.hl_play_center_zone.setSpacing(4)
        self.hl_play_center_zone.setObjectName(u"hl_play_center_zone")
        self.hl_play_center_zone.setContentsMargins(0, 0, 0, 0)
        self.btn_first_frame = QPushButton(self.centralwidget)
        self.btn_first_frame.setObjectName(u"btn_first_frame")

        self.hl_play_center_zone.addWidget(self.btn_first_frame)

        self.btn_prev_frame = QPushButton(self.centralwidget)
        self.btn_prev_frame.setObjectName(u"btn_prev_frame")

        self.hl_play_center_zone.addWidget(self.btn_prev_frame)

        self.btn_play_reverse = QPushButton(self.centralwidget)
        self.btn_play_reverse.setObjectName(u"btn_play_reverse")
        self.btn_play_reverse.setMinimumSize(QSize(39, 39))
        self.btn_play_reverse.setMaximumSize(QSize(39, 39))
        self.btn_play_reverse.setIconSize(QSize(26, 26))
        self.btn_play_reverse.setCheckable(True)

        self.hl_play_center_zone.addWidget(self.btn_play_reverse)

        self.btn_play = QPushButton(self.centralwidget)
        self.btn_play.setObjectName(u"btn_play")
        self.btn_play.setMinimumSize(QSize(39, 39))
        self.btn_play.setMaximumSize(QSize(39, 39))
        self.btn_play.setIconSize(QSize(26, 26))
        self.btn_play.setCheckable(True)

        self.hl_play_center_zone.addWidget(self.btn_play)

        self.btn_next_frame = QPushButton(self.centralwidget)
        self.btn_next_frame.setObjectName(u"btn_next_frame")

        self.hl_play_center_zone.addWidget(self.btn_next_frame)

        self.btn_last_frame = QPushButton(self.centralwidget)
        self.btn_last_frame.setObjectName(u"btn_last_frame")

        self.hl_play_center_zone.addWidget(self.btn_last_frame)

        self.btn_play_loop = QPushButton(self.centralwidget)
        self.btn_play_loop.setObjectName(u"btn_play_loop")
        self.btn_play_loop.setCheckable(True)

        self.hl_play_center_zone.addWidget(self.btn_play_loop)


        self.hl_play_row.addLayout(self.hl_play_center_zone)

        self.hl_play_right_zone = QHBoxLayout()
        self.hl_play_right_zone.setSpacing(6)
        self.hl_play_right_zone.setObjectName(u"hl_play_right_zone")
        self.hl_play_right_zone.setContentsMargins(0, 0, 0, 0)
        self.spacer_play_right_push = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.hl_play_right_zone.addItem(self.spacer_play_right_push)

        self.btn_run = QPushButton(self.centralwidget)
        self.btn_run.setObjectName(u"btn_run")
        self.btn_run.setMinimumSize(QSize(190, 34))
        self.btn_run.setMaximumSize(QSize(190, 34))
        self.btn_run.setStyleSheet(u"QPushButton {\n"
"    background-color: #0f5a33;\n"
"    border: 1px solid #29a56a;\n"
"    color: #f0f8f4;\n"
"    font-size: 14px;\n"
"    border-radius: 10px;\n"
"}\n"
"QPushButton:hover { background-color: #1f8f58; }\n"
"QPushButton:pressed { background-color: #0c4527; }\n"
"QPushButton:disabled {\n"
"    background-color: #15261c;\n"
"    border: 1px solid #233529;\n"
"    color: #4b6a57;\n"
"}")

        self.hl_play_right_zone.addWidget(self.btn_run)

        self.btn_stop = QPushButton(self.centralwidget)
        self.btn_stop.setObjectName(u"btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setMinimumSize(QSize(104, 34))
        self.btn_stop.setMaximumSize(QSize(104, 34))
        self.btn_stop.setStyleSheet(u"QPushButton {\n"
"    background-color: #5a1f22;\n"
"    border: 1px solid #a03c41;\n"
"    color: #f0f0f0;\n"
"    border-radius: 9px;\n"
"}\n"
"QPushButton:hover { background-color: #8f3238; }\n"
"QPushButton:pressed { background-color: #431317; }\n"
"QPushButton:disabled {\n"
"    background-color: #24171a;\n"
"    border: 1px solid #3b272b;\n"
"    color: #6a4a4a;\n"
"}")

        self.hl_play_right_zone.addWidget(self.btn_stop)

        self.btn_save_result = QPushButton(self.centralwidget)
        self.btn_save_result.setObjectName(u"btn_save_result")
        self.btn_save_result.setMinimumSize(QSize(158, 34))
        self.btn_save_result.setMaximumSize(QSize(158, 34))

        self.hl_play_right_zone.addWidget(self.btn_save_result)


        self.hl_play_row.addLayout(self.hl_play_right_zone)


        self.vl_center_work.addLayout(self.hl_play_row)

        self.progress_bar = QProgressBar(self.centralwidget)
        self.progress_bar.setObjectName(u"progress_bar")
        self.progress_bar.setMinimumSize(QSize(0, 15))
        self.progress_bar.setMaximumSize(QSize(16777215, 13))
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        self.vl_center_work.addWidget(self.progress_bar)


        self.hl_control_row.addLayout(self.vl_center_work)

        self.vl_sidebar = QVBoxLayout()
        self.vl_sidebar.setSpacing(6)
        self.vl_sidebar.setObjectName(u"vl_sidebar")
        self.vl_sidebar.setContentsMargins(8, 0, 0, 2)
        self.lbl_sec_actions = QLabel(self.centralwidget)
        self.lbl_sec_actions.setObjectName(u"lbl_sec_actions")
        self.lbl_sec_actions.setVisible(False)
        self.lbl_sec_actions.setStyleSheet(u"color: #666666; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        self.lbl_sec_actions.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.vl_sidebar.addWidget(self.lbl_sec_actions)

        self.sp_sidebar_bottom = QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.vl_sidebar.addItem(self.sp_sidebar_bottom)


        self.hl_control_row.addLayout(self.vl_sidebar)


        self.vl_main.addLayout(self.hl_control_row)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QStatusBar(MainWindow)
        self.statusBar.setObjectName(u"statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"KeyFlow Studio", None))
        self.lbl_brand.setText(QCoreApplication.translate("MainWindow", u"KEYFLOW STUDIO", None))
        self.lbl_brand_badge.setText(QCoreApplication.translate("MainWindow", u"0.0", None))
        self.lbl_file_path.setText(QCoreApplication.translate("MainWindow", u"\u0424\u0430\u0439\u043b \u043d\u0435 \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043d", None))
#if QT_CONFIG(tooltip)
        self.btn_settings.setToolTip(QCoreApplication.translate("MainWindow", u"\u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438 \u043f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u044f", None))
#endif // QT_CONFIG(tooltip)
        self.btn_settings.setText("")
        self.lbl_input_title.setText(QCoreApplication.translate("MainWindow", u"INPUT PREVIEW", None))
        self.input_video_label.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u0435 \u0432\u0438\u0434\u0435\u043e \u0438\u043b\u0438 \u0444\u043e\u0442\u043e", None))
#if QT_CONFIG(tooltip)
        self.btn_preview_foreground.setToolTip(QCoreApplication.translate("MainWindow", u"Display Gamma", None))
#endif // QT_CONFIG(tooltip)
        self.btn_preview_foreground.setText("")
#if QT_CONFIG(tooltip)
        self.btn_preview_alpha.setToolTip(QCoreApplication.translate("MainWindow", u"Linear Preview", None))
#endif // QT_CONFIG(tooltip)
        self.btn_preview_alpha.setText("")
        self.lbl_output_title.setText(QCoreApplication.translate("MainWindow", u"OUTPUT PREVIEW", None))
        self.btn_split_view.setText("")
        self.output_video_label.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u043f\u043e\u044f\u0432\u0438\u0442\u0441\u044f \u0437\u0434\u0435\u0441\u044c", None))
        self.lbl_frame_info.setText(QCoreApplication.translate("MainWindow", u"0000 / 0000", None))
        self.combo_playback_presets.setItemText(0, QCoreApplication.translate("MainWindow", u"Preset: Default", None))

        self.lbl_start_frame.setText(QCoreApplication.translate("MainWindow", u"Start:", None))
        self.lbl_end_frame.setText(QCoreApplication.translate("MainWindow", u"End:", None))
        self.spin_end_frame.setSpecialValueText(QCoreApplication.translate("MainWindow", u"Auto", None))
        self.lbl_num_frames.setText(QCoreApplication.translate("MainWindow", u"Frames:", None))
#if QT_CONFIG(tooltip)
        self.spin_num_frames.setToolTip(QCoreApplication.translate("MainWindow", u"0 = \u0432\u0441\u0435 \u043a\u0430\u0434\u0440\u044b \u0432\u0438\u0434\u0435\u043e", None))
#endif // QT_CONFIG(tooltip)
        self.spin_num_frames.setSpecialValueText(QCoreApplication.translate("MainWindow", u"\u0412\u0441\u0435", None))
#if QT_CONFIG(tooltip)
        self.btn_first_frame.setToolTip(QCoreApplication.translate("MainWindow", u"\u041f\u0435\u0440\u0432\u044b\u0439 \u043a\u0430\u0434\u0440", None))
#endif // QT_CONFIG(tooltip)
        self.btn_first_frame.setText(QCoreApplication.translate("MainWindow", u"\u23ee", None))
#if QT_CONFIG(tooltip)
        self.btn_prev_frame.setToolTip(QCoreApplication.translate("MainWindow", u"\u041f\u0440\u0435\u0434\u044b\u0434\u0443\u0449\u0438\u0439 \u043a\u0430\u0434\u0440", None))
#endif // QT_CONFIG(tooltip)
        self.btn_prev_frame.setText(QCoreApplication.translate("MainWindow", u"\u25c0", None))
#if QT_CONFIG(tooltip)
        self.btn_play_reverse.setToolTip(QCoreApplication.translate("MainWindow", u"\u0412\u043e\u0441\u043f\u0440\u043e\u0438\u0437\u0432\u0435\u0441\u0442\u0438 \u043d\u0430\u0437\u0430\u0434", None))
#endif // QT_CONFIG(tooltip)
        self.btn_play_reverse.setText(QCoreApplication.translate("MainWindow", u"\u25c0", None))
#if QT_CONFIG(tooltip)
        self.btn_play.setToolTip(QCoreApplication.translate("MainWindow", u"\u0412\u043e\u0441\u043f\u0440\u043e\u0438\u0437\u0432\u0435\u0441\u0442\u0438 / \u041f\u0430\u0443\u0437\u0430", None))
#endif // QT_CONFIG(tooltip)
        self.btn_play.setText(QCoreApplication.translate("MainWindow", u"\u25b6", None))
#if QT_CONFIG(tooltip)
        self.btn_next_frame.setToolTip(QCoreApplication.translate("MainWindow", u"\u0421\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0439 \u043a\u0430\u0434\u0440", None))
#endif // QT_CONFIG(tooltip)
        self.btn_next_frame.setText(QCoreApplication.translate("MainWindow", u"\u25b6", None))
#if QT_CONFIG(tooltip)
        self.btn_last_frame.setToolTip(QCoreApplication.translate("MainWindow", u"\u041f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0439 \u043a\u0430\u0434\u0440", None))
#endif // QT_CONFIG(tooltip)
        self.btn_last_frame.setText(QCoreApplication.translate("MainWindow", u"\u23ed", None))
#if QT_CONFIG(tooltip)
        self.btn_play_loop.setToolTip(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0446\u0438\u043a\u043b\u0438\u0442\u044c \u0432\u043e\u0441\u043f\u0440\u043e\u0438\u0437\u0432\u0435\u0434\u0435\u043d\u0438\u0435", None))
#endif // QT_CONFIG(tooltip)
        self.btn_play_loop.setText(QCoreApplication.translate("MainWindow", u"\u21bb", None))
#if QT_CONFIG(tooltip)
        self.btn_run.setToolTip(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c \u0438\u043d\u0444\u0435\u0440\u0435\u043d\u0441 MatAnyone2", None))
#endif // QT_CONFIG(tooltip)
        self.btn_run.setText(QCoreApplication.translate("MainWindow", u"Start Processing", None))
        self.btn_stop.setText(QCoreApplication.translate("MainWindow", u"Stop", None))
#if QT_CONFIG(tooltip)
        self.btn_save_result.setToolTip(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u043d\u0430 \u0434\u0438\u0441\u043a", None))
#endif // QT_CONFIG(tooltip)
        self.btn_save_result.setText(QCoreApplication.translate("MainWindow", u"Save Result", None))
        self.lbl_sec_actions.setText(QCoreApplication.translate("MainWindow", u"ACTIONS", None))
    # retranslateUi

