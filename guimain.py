from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen
import os


global vsrc
vsrc = ObjectProperty(None)

class MainWindow(Screen):
    pass


class SecondWindow(Screen):
	
	def analyze(self):
		vsrc = self.vsrc.text
		os.system('activate project2.7 && python facedetect.py -v '+ vsrc + '&& deactivate')
		return vsrc


class ThirdWindow(Screen):
	imgloc = ObjectProperty(None)
	def search(self):
		imgloc = self.imgloc.text
		os.system('activate project2.7 && python search.py '+ imgloc +'&& deactivate')


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("my.kv")


class MyMainApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    MyMainApp().run()