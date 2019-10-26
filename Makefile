#CC = clang-3.6
#CXX = clang++-3.6
CC = gcc
CXX = g++

INCLUDES = ./

#CFLAGS = -O3 -fopenmp -Wall -Wno-parentheses
CFLAGS += -g -Wall -Wno-parentheses
CFLAGS += -I$(INCLUDES)

CPPFLAGS = $(CFLAGS) -std=c++11
#CPPFLAGS += `pkg-config opencv --cflags`
CPPFLAGS += -I /home/user/opencv-3.2.0/include/
CPPFLAGS += -I ./include/

#LDFLAGS += `pkg-config opencv --libs`
LDFLAGS += -L /home/user/opencv-3.2.0/lib/ -lopencv_dnn -lopencv_ml -lopencv_objdetect -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_imgproc -lopencv_flann -lopencv_core
#LDFLAGS += -lopencv_core -lopencv_imgcodecs -lopencv_calib3d -lopencv_imgproc
LDFLAGS += -lboost_system -lboost_filesystem

CSRCS =
#CPPSRCS = calibration.cpp
#CPPSRCS = app_fisheye_calibrate.cpp
#CPPSRCS = omni_calibration.cpp
CPPSRCS = main.cpp
CPPSRCS += ./src/omnidir.cpp

OBJS = $(CSRCS:.c=.o) $(CPPSRCS:.cpp=.o)

TARGET = a.out

.PHONY: depend clean

all: $(TARGET) imagelist_creator

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -o $@ -c $<

imagelist_creator: imagelist_creator.cpp
	$(CXX) -o $@ $< $(CPPFLAGS) $(LDFLAGS)

clean:
	$(RM) $(OBJS) $(EXTRAS) $(TARGET) imagelist_creator

distclean: clean
	$(RM) *~ .depend

#depend: $(CSRCS) $(CPPSRCS)
#	makedepend $(INCLUDES) $^

depend: .depend

.depend: $(CSRCS) $(CPPSRCS) imagelist_creator.cpp
	$(RM) ./.depend
	$(CXX) $(CPPFLAGS) -MM $^ >> ./.depend

include .depend
