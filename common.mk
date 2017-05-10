SHELL=/bin/sh -ue

CFLAGS   += -O3
CXXFLAGS += -O3

ifdef OUTPUT
CPPFLAGS += -DOUTPUT
endif

ifdef DEBUG
CFLAGS   += -g
CXXFLAGS += -g
endif

ROOT:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ifeq (exists, $(shell [ -e $(ROOT)/Make.user ] && echo exists ))
include $(ROOT)/Make.user
endif
