# Vendored third-party jars

## bounce.jar

The Bounce library (`org.bounce`, by Edwin Dankert — https://bounce.sourceforge.net).
Weka's `WekaPackageManager` static initializer references
`org.bounce.net.DefaultAuthenticator`, so Weka's J48 fails with a
`NoClassDefFoundError` if this jar is not on the classpath. The Maven Central
`weka-stable-3.8.6.jar` does not bundle it, so it is vendored here (208 KB) and
placed on `WEKA_JAR` alongside the Weka jar by the CI workflows.

- sha256: `bffff1505335c02256b7ab2ccffbe4aa4d3ac9fe14c17557809b7c9d99d666ca`
- Same jar distributed with the Weka 3.8.6 release. Bounce is distributed under
  the Apache License 2.0; confirm and add the upstream LICENSE text before any
  formal release if required.
