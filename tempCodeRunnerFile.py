while True:

#     _, frame = capture.read()
#     if frame is None:
#         print("End of stream")
#         break

#     inputImage = format_yolov5(frame)
#     outs = detect(inputImage, net)

#     class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

#     frame_count += 1
#     total_frames += 1

#     for (classid, confidence, box) in zip(class_ids, confidences, boxes):
#          color = colors[int(classid) % len(colors)]
#          cv2.rectangle(frame, box, color, 2)
#          cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
#          cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

#     if frame_count >= 30:
#         end = time.time_ns()
#         fps = 1000000000 * frame_count / (end - start)
#         frame_count = 0
#         start = time.time_ns()
    
#     if fps > 0:
#         fps_label = "FPS: %.2f" % fps
#         cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow("output", frame)

#     if cv2.waitKey(1) > -1:
#         print("finished by user")
#         break

# print("Total frames: " + str(total_frames))