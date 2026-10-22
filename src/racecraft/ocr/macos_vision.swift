import Foundation
import Vision

guard CommandLine.arguments.count == 2 else {
    fputs("Usage: macos_vision.swift IMAGE_PATH\n", stderr)
    exit(2)
}

let url = URL(fileURLWithPath: CommandLine.arguments[1])
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
request.minimumTextHeight = 0.008

let handler = VNImageRequestHandler(url: url, options: [:])
try handler.perform([request])
let observations = (request.results ?? []).compactMap { observation -> [String: Any]? in
    guard let candidate = observation.topCandidates(1).first else { return nil }
    let box = observation.boundingBox
    return [
        "text": candidate.string,
        "x": box.origin.x,
        "y": box.origin.y,
        "width": box.width,
        "height": box.height
    ]
}
let data = try JSONSerialization.data(withJSONObject: observations)
FileHandle.standardOutput.write(data)
