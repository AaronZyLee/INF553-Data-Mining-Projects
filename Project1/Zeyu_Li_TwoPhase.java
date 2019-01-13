// Two phase matrix multiplication in Hadoop MapReduce
// Template file for homework #1 - INF 553 - Spring 2017
// - Wensheng Wu

import java.io.IOException;

// add your import statement here if needed
// you can only import packages from java.*;

import org.apache.hadoop.conf.Configuration;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;

import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.util.ArrayList;

public class TwoPhase {

    // mapper for processing entries of matrix A
    public static class PhaseOneMapperA extends Mapper<LongWritable, Text, Text, Text> {
	
        private Text outKey = new Text();
        private Text outVal = new Text();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            
            String[] lines = value.toString().split("\n");
            for (String s: lines) {
                String temp[] = s.split(",");
                context.write(new Text(temp[1]), new Text(new String("A," + temp[0] + "," + temp[2])));
            }
        }
    }

    // mapper for processing entries of matrix B
    public static class PhaseOneMapperB extends Mapper<LongWritable, Text, Text, Text> {
	
        private Text outKey = new Text();
        private Text outVal = new Text();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
	    
            String[] lines = value.toString().split("\n");
            for (String s: lines) {
                String temp[] = s.split(",");
                context.write(new Text(temp[0]), new Text(new String("B," + temp[1] + "," + temp[2])));
            }
        }
    }

    public static class PhaseOneReducer extends Reducer<Text, Text, Text, Text> {

        private Text outKey = new Text();
        private Text outVal = new Text();

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            
            /*
            for (Text txt: values) {
                String[] temp1 = txt.toString().split(",");
                if (temp1[0].equals("A")) {
                    
                    for (Text txt2: values) {
                        String[] temp2 = txt2.toString().split(",");
                        if (temp2[0].equals("B")) {
                            context.write(new Text(new String(temp1[1] + "," + temp2[1])), new Text(String.valueOf(Integer.parseInt(temp1[2]) * Integer.parseInt(temp2[2]))));
                        }
                    }
                }
            }
            */
            ArrayList<String[]> alist = new ArrayList<>();
            ArrayList<String[]> blist = new ArrayList<>();
            for (Text txt: values) {
                String[] temp1 = txt.toString().split(",");
                if (temp1[0].equals("A")) {
                    alist.add(temp1);
                }
                if (temp1[0].equals("B")) {
                    blist.add(temp1);
                }
            }
            
            for (String[] s1: alist) {
                for (String[] s2: blist) {
                    context.write(new Text(new String(s1[1] + "," + s2[1])), new Text(String.valueOf(Integer.parseInt(s1[2]) * Integer.parseInt(s2[2]))));
                }
            }

            
            
        }
    }

    public static class PhaseTwoMapper extends Mapper<Text, Text, Text, Text> {
	
        private Text outKey = new Text();
        private Text outVal = new Text();

        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {

            context.write(key, value);

        }
    }

    public static class PhaseTwoReducer extends Reducer<Text, Text, Text, Text> {
	
        private Text outKey = new Text();
        private Text outVal = new Text();

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
	 
            int sum = 0;
            for (Text txt: values) {
                sum += Integer.parseInt(txt.toString());
            }
            context.write(key,new Text(String.valueOf(sum)));

        }
    }


    public static void main(String[] args) throws Exception {
	Configuration conf = new Configuration();

	Job jobOne = Job.getInstance(conf, "phase one");

	jobOne.setJarByClass(TwoPhase.class);

	jobOne.setOutputKeyClass(Text.class);
	jobOne.setOutputValueClass(Text.class);

	jobOne.setReducerClass(PhaseOneReducer.class);

	MultipleInputs.addInputPath(jobOne,
				    new Path(args[0]),
				    TextInputFormat.class,
				    PhaseOneMapperA.class);

	MultipleInputs.addInputPath(jobOne,
				    new Path(args[1]),
				    TextInputFormat.class,
				    PhaseOneMapperB.class);

	Path tempDir = new Path("temp");

	FileOutputFormat.setOutputPath(jobOne, tempDir);
	jobOne.waitForCompletion(true);


	// job two
	Job jobTwo = Job.getInstance(conf, "phase two");
	

	jobTwo.setJarByClass(TwoPhase.class);

	jobTwo.setOutputKeyClass(Text.class);
	jobTwo.setOutputValueClass(Text.class);

	jobTwo.setMapperClass(PhaseTwoMapper.class);
	jobTwo.setReducerClass(PhaseTwoReducer.class);

	jobTwo.setInputFormatClass(KeyValueTextInputFormat.class);

	FileInputFormat.setInputPaths(jobTwo, tempDir);
	FileOutputFormat.setOutputPath(jobTwo, new Path(args[2]));
	
	jobTwo.waitForCompletion(true);
	
	FileSystem.get(conf).delete(tempDir, true);
	
    }
}

