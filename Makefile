#CC = clang-3.6
#CXX = clang++-3.6
CC = gcc
CXX = g++

INCLUDES = ./

#CFLAGS = -O3 -fopenmp -Wall -Wno-parentheses
CFLAGS += -g -Wall -Wno-parentheses
CFLAGS += -I$(INCLUDES)

CPPFLAGS = $(CFLAGS) -std=c++11
CPPFLAGS += `pkg-config opencv --cflags`

LDFLAGS += `pkg-config opencv --libs`
LDFLAGS += -lboost_system -lboost_filesystem

CSRCS =
CPPSRCS = main.cpp

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
