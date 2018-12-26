pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile'
            dir '.docker'
            args '--runtime nvidia -v /tmp/adept_logs:/tmp/adept_logs'
        }
    }

    stages {
        stage('Build') {
            steps {
                echo 'Building..'
                sh 'nvidia-smi'
                sh 'pip3 install .[all]'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
                sh 'pytest --verbose --junit-xml test_reports/results.xml'
            }
            post {
                always {
                    junit
                }
            }
        }
    }
}
