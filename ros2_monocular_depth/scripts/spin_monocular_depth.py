import ros2_utils.node as utils_node

from ros2_monocular_depth.ros import NodeMonocularDepth


def main():
    utils_node.start_and_spin_node(NodeMonocularDepth)


if __name__ == "__main__":
    main()
