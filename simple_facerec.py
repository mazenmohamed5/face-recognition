{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "69a097f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import face_recognition\n",
    "import os\n",
    "import cv2\n",
    "import numpy as np \n",
    "import glob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "18bc9797",
   "metadata": {},
   "outputs": [],
   "source": [
    "class SimpleFacerec:\n",
    "    def __init__(self):\n",
    "        self.known_face_encodings = []\n",
    "        self.known_face_names = []\n",
    "        \n",
    "        #Resize frame for faster speed\n",
    "        self.frame_resizing=0.25\n",
    "        \n",
    "    def load_encoding_images(self,images_path):\n",
    "        \"\"\"\n",
    "        Load encoding images from path\n",
    "        :param images_path\n",
    "        :return:\n",
    "        \"\"\"\n",
    "        #Load images\n",
    "        image_path=glob.glob(os.path.join(images_path,\"*.*\"))\n",
    "        \n",
    "        print(\"{} encoding images found.\".format(len(image_path)))\n",
    "        \n",
    "         #store image encoding and names\n",
    "        for img_pth in image_path:\n",
    "            img=cv2.imread(img_pth)\n",
    "            rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)\n",
    "            \n",
    "            #Get file name only from path\n",
    "            basename=os.path.basename(img_pth)\n",
    "            (filename,ext)=os.path.splitext(basename)\n",
    "            #Get encoding\n",
    "            img_encoding=face_recognition.face_encodings(rgb_img)[0]\n",
    "            \n",
    "            #Store file name and file encoding\n",
    "            self.known_face_encodings.append(img_encoding)\n",
    "            self.known_face_names.append(filename)\n",
    "        print('encoding image loaded')    \n",
    "    \n",
    "    def detect_known_faces(self,frame):\n",
    "        small_frame=cv2.resize(frame,(0,0), fx=self.frame_resizing, fy=self.frame_resizing)\n",
    "        #find face and encoding in the current frame of video\n",
    "        #convert the img fror BGR to RGB\n",
    "        rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)\n",
    "        face_locations=face_recognition.face_locations(rgb_small_frame)\n",
    "        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)\n",
    "        \n",
    "        face_names = []\n",
    "        for face_encoding in face_encodings:\n",
    "            #see if the face is match for the known face(s)\n",
    "            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)\n",
    "            name = 'Unknown'\n",
    "            \n",
    "            #use the knonw face with the smallest distance\n",
    "            face_distance = face_recognition.face_distance(self.known_face_encodings,face_encoding)\n",
    "            best_match_index = np.argmin(face_distance)\n",
    "            if matches [best_match_index]:\n",
    "                name = self.known_face_names[best_match_index]\n",
    "            face_names.append(name)\n",
    "            \n",
    "        # convert to numpy array to adjust coordinates with frame resizing\n",
    "        face_locations = np.array(face_locations)\n",
    "        face_locations = face_locations/self.frame_resizing\n",
    "        return face_locations.astype(int), face_names\n",
    "             \n",
    "        \n",
    "       \n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8842dcbf",
   "metadata": {
    "scrolled": True
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4 encoding images found.\n",
      "encoding image loaded\n"
     ]
    }
   ],
   "source": [
    "#encode faces from folder\n",
    "sfr=SimpleFacerec()\n",
    "sfr.load_encoding_images('images')\n",
    "\n",
    "#load camera\n",
    "cap=cv2.VideoCapture(0)\n",
    "\n",
    "while True:\n",
    "    ret,frame = cap.read()\n",
    "    \n",
    "    # Detect Faces\n",
    "    face_locations,face_names = sfr.detect_known_faces(frame)\n",
    "    for face_loc, name in zip(face_locations,face_names):\n",
    "        y1,x2,y2,x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]\n",
    "        \n",
    "        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)\n",
    "        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,200), 4)\n",
    "    \n",
    "    cv2.imshow('Frame',frame)\n",
    "    key=cv2.waitKey(1)\n",
    "    if key==27:\n",
    "        break\n",
    "        \n",
    "        \n",
    "cap.release()\n",
    "cv2.destroyAllWindows()\n",
    "    \n",
    "    \n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d8837b95",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc9c37cf",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a94dec72",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20e4b9c6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bec07d95",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e1d3c012",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "ML",
   "language": "python",
   "name": "ml"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
