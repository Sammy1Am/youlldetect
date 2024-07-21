# You'll Detect

The goal of this project is to provide a lightweight, performant service for object detection.  The initial goal is to run YOLOv8 on a Coral TPM to provide detection for Frigate, but long-term should have support for any other Ultralytics supported model, other acceleration options, and with various convenient APIs.

To that end, the `detect_cpai` endpoint will emulate (as much as is possible and/or as much as is necessary for Frigate) the [CodeProject.AI object detection API](https://www.codeproject.com/AI/docs/api/api_reference.html#object-detection) since Frigate already has support for this.  Eventually a more optimized Frigate-specific API should be implemented (possibly using WebSockets for performance), but this will likely require implementation on the Frigate side of things as well.

Contributions to enhance performance and reliability are greatly appreciated as I'm still a bit green in the ML/object detection space.