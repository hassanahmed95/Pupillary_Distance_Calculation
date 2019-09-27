from lxml import etree
import os, glob, natsort

def getBboxesFromFile(filename):
	#This function assume that the annotations have been done using labelimg tool.
	tree = etree.parse(filename);
	root = tree.getroot();

	bboxes = [];

	for child in root:
		if (child.tag == 'object'):
			xmin = int(child[4][0].text);
			ymin = int(child[4][1].text);
			xmax = int(child[4][2].text);
			ymax = int(child[4][3].text);

			bbox = (xmin, ymin, xmax, ymax);

			bboxes.append(bbox);

	return bboxes;



if __name__ == '__main__':

	TEST_FILE_OFFSET = 5
	path_to_input_xmls = 'XML FILES';


	path_to_input_xmls = "/home/hassanahmed/PycharmProjects/pupil_distance/data/My_New_data/MY_DATA/XML_Flies"

	xml_files = natsort.natsorted( glob.glob(os.path.join(path_to_input_xmls, '*.xml')) );

	training_xml_name = 'training.xml';
	testing_xml_name = 'testing.xml';


	dataset_train = etree.Element('dataset')
	dataset_test = etree.Element('dataset')

	doc_train = etree.ElementTree(dataset_train)
	doc_test = etree.ElementTree(dataset_test)

	name_train = etree.SubElement(dataset_train,'name');
	name_train.text = 'Training Card'

	name_test = etree.SubElement(dataset_test,'name');
	name_test.text = 'Testing Card'

	comment_train = etree.SubElement(dataset_train,'comment');
	comment_train.text = 'Dataset for card detection'

	comment_test = etree.SubElement(dataset_test,'comment');
	comment_test.text = 'Dataset for card testing'

	images_train = etree.SubElement(dataset_train,'images');
	images_test = etree.SubElement(dataset_test,'images');

	for i, xml_file in enumerate(xml_files):
		bboxes = getBboxesFromFile(xml_file);

		name_only = os.path.basename(xml_file);
		jpg_name = str(name_only[:-4]) + '.jpg';

		if (i+1) % TEST_FILE_OFFSET == 0:
			image = etree.SubElement(images_test, 'image', attrib = { 'file' : jpg_name });
		else:
			image = etree.SubElement(images_train, 'image', attrib = { 'file' : jpg_name });

		for bbox in bboxes:
			width = bbox[2] - bbox[0] + 1;
			height = bbox[3] - bbox[1] + 1;

			att = { 'top' : str(bbox[1]), 'left' : str(bbox[0]), 'width' : str(width), 'height' : str(height) };
			box = etree.SubElement(image, 'box' , attrib = att);


	pretty_xml_train = etree.tostring(doc_train, pretty_print=True);
	outFile_train = open(training_xml_name, 'wb')
	outFile_train.write(pretty_xml_train);	

	pretty_xml_test = etree.tostring(doc_test, pretty_print=True);
	outFile_test = open(testing_xml_name, 'wb')
	outFile_test.write(pretty_xml_test);	
