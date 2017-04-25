import matplotlib.pyplot as plt


def drawComplex(data, ph, axes=[-6, 8, -6, 6]):
    plt.clf()
    plt.axis(axes)  # axes = [x1, x2, y1, y2]
    plt.scatter(data[:, 0], data[:, 1])  # plotting just for clarity
    for i, txt in enumerate(data):
        plt.annotate(i, (data[i][0] + 0.05, data[i][1]))  # add labels

    # add lines for edges
    for edge in [e for e in ph.ripsComplex if len(e) == 2]:
        # print(edge)
        pt1, pt2 = [data[pt] for pt in [n for n in edge]]
        # plt.gca().add_line(plt.Line2D(pt1,pt2))
        line = plt.Polygon([pt1, pt2], closed=None, fill=None, edgecolor='r')
        plt.gca().add_line(line)

    # add triangles
    for triangle in [t for t in ph.ripsComplex if len(t) == 3]:
        pt1, pt2, pt3 = [data[pt] for pt in [n for n in triangle]]
        line = plt.Polygon([pt1, pt2, pt3], closed=False,
                           color="blue", alpha=0.3, fill=True, edgecolor=None)
        plt.gca().add_line(line)
    plt.show()


def graph_barcode(data, ph, homology_group=0):
    persistence = ph.transform(data)
    # this function just produces the barcode graph for each homology group
    xstart = [s[1][0] for s in persistence if s[0] == homology_group]
    xstop = [s[1][1] for s in persistence if s[0] == homology_group]
    y = [0.1 * x + 0.1 for x in range(len(xstart))]
    plt.hlines(y, xstart, xstop, color='b', lw=4)
    # Setup the plot
    ax = plt.gca()
    plt.ylim(0, max(y) + 0.1)
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    plt.xlabel('epsilon')
    plt.ylabel("Betti dim %s" % (homology_group,))
    plt.show()
