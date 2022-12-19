def test_example_magic_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    # layer = viewer.add_image(np.random.random((100, 100)))

    # this time, our widget will be a MagicFactory or FunctionGui instance
    # my_widget = wizard_widget()

    # if we "call" this object, it'll execute our function
    # my_widget(viewer,)

    # read captured output and check that it's as we expected
    # captured = capsys.readouterr()


#    assert captured.out == f"you have selected {layer}\n"
