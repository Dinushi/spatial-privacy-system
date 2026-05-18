from typing import Any, Dict, List


def print_available_objects(metadata_plain: Dict[str, Any]) -> None:
    print("\nAvailable hidden objects:")
    for obj in metadata_plain.get("objects", []):
        print(
            f"- object_id={obj.get('object_id')} | "
            f"label={obj.get('label')} | "
            f"class_id={obj.get('class_id')} | "
            f"regions={obj.get('region_ids')}"
        )


def find_matching_objects(metadata_plain: Dict[str, Any], request_text: str) -> List[Dict[str, Any]]:
    """
    Simple keyword matching.
    Example request: 'please reveal brand of cream bottle'
    """
    request_lower = request_text.lower()
    matches = []

    for obj in metadata_plain.get("objects", []):
        label = str(obj.get("label", "")).lower()
        class_id = str(obj.get("class_id", "")).lower()

        if label in request_lower or any(word in label for word in request_lower.split()):
            matches.append(obj)
        elif class_id and class_id in request_lower:
            matches.append(obj)

    return matches