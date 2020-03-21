#Made By XrangerCY19
import os
import sys
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button


global vsrc  				#Video Source
vsrc = ObjectProperty(None)

class MainWindow(Screen):	#First Window
    pass

class SecondWindow(Screen):	#Second Window
	
	def analyze(self):		#Function to analyze/scan and detect faces in the Video 
		vsrc = self.vsrc.text
		os.system('activate project2.7 && python facedetect.py -v '+ vsrc + '&& deactivate')
		return vsrc



class ThirdWindow(Screen):	#Third Window
	imgloc = ObjectProperty(None) 	#Image Location. 
	def search(self):				#function to search the image in the DB
		imgloc = self.imgloc.text
		os.system('activate project2.7 && python search.py '+ imgloc +'&& deactivate')

		with open('file.txt', 'r') as myfile:	#To read the data from file.txt
  			data = myfile.read()

		layout = GridLayout(cols=1, padding=10)	#Popup widget
		popupLabel  = Label(text  = data)
		layout.add_widget(popupLabel)  
		Popup(title='Result.',content=layout,size_hint =(None, None), size =(400, 350)).open()  

#Made By XrangerCY19

class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("my.kv")		#Load UI from my.kv


class MyMainApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    MyMainApp().run()

#Made By XrangerCY19